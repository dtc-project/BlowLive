#!/usr/bin/env python3

import os
import time
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt, stft
from scipy.fftpack import dct
from gammatone.gtgram import gtgram  # pip install gammatone
from typing import List, Dict, Tuple


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = "LivenessDetection_Dataset"     # dataset/<user>/{legitimate,attacker}/*.wav
FS = 48000 # Sampling rate
F0 = 20000 # Emitted ultrasonic signal
LOW = 19500
HIGH = 20500
NUM_CEPS = 13
MAX_FRAMES = 300
N_THRESHOLD_STEPS = 200      # resolution for threshold search
SPLIT_RATIO = 1            # fraction of legitimate/attacker used for training (0<r<=1)


# ============================================================
# AUDIO / FEATURE EXTRACTION
# ============================================================

def bandpass_20k(x: np.ndarray, fs: float = FS) -> np.ndarray:
    """
    4th-order bandpass filter around 20 kHz using SOS form.
    """
    sos = butter(4, [LOW, HIGH], btype="bandpass", fs=fs, output="sos")
    return sosfilt(sos, x)


def extract_gfcc_sequence(wav_path: str) -> np.ndarray:
    """
    Extract GFCC sequence (T x NUM_CEPS) from audio, band-passed around 20 kHz,
    using gammatone spectrogram + DCT.
    Padded/truncated to MAX_FRAMES.
    """
    fs_audio, x = wavfile.read(wav_path)
    x = x.astype(float)

    if fs_audio != FS:
        raise ValueError(f"[GFCC] Expected {FS} Hz, got {fs_audio} Hz for {wav_path}")

    x_bp = bandpass_20k(x, fs=FS)

    # Gammatone spectrogram: (num_channels, time_frames)
    gt_spec = gtgram(x_bp, FS, 0.025, 0.010, 32, 50.0)
    log_spec = np.log(gt_spec + 1e-12)
    ceps = dct(log_spec, axis=0, norm="ortho")

    gfcc = ceps[:NUM_CEPS, :].T  # (T, NUM_CEPS)

    # pad / truncate
    T = gfcc.shape[0]
    if T >= MAX_FRAMES:
        gfcc_fixed = gfcc[:MAX_FRAMES, :]
    else:
        pad = np.zeros((MAX_FRAMES - T, NUM_CEPS))
        gfcc_fixed = np.vstack([gfcc, pad])
    return gfcc_fixed


def extract_doppler_features(wav_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Doppler extraction:
      - bandpass around 20 kHz
      - STFT
      - track peak around 20 kHz per frame
      - Doppler shift f_D(t) and envelope env(t)
    """
    fs_audio, x = wavfile.read(wav_path)
    x = x.astype(float)

    if fs_audio != FS:
        raise ValueError(f"[Doppler] Expected {FS} Hz, got {fs_audio} Hz for {wav_path}")

    x_bp = bandpass_20k(x, fs=FS)

    f, t, Z = stft(x_bp, fs=FS, nperseg=2048, noverlap=1024)
    band = (f >= LOW) & (f <= HIGH)
    f_band = f[band]
    Z_band = Z[band, :]   # (F_band, T)

    mag = np.abs(Z_band)
    peak_idx = np.argmax(mag, axis=0)
    f_peak = f_band[peak_idx]

    # doppler shift around 20 kHz
    f_D = f_peak - F0

    # doppler envelope (RMS of magnitude across band)
    env = np.sqrt(np.mean(mag ** 2, axis=0))

    return f_D, env


def compute_hybrid_score(gfcc_energy: float, doppler_rms: float, doppler_max: float) -> float:
    """
    Hybrid score = weighted combination of GFCC energy and Doppler stats.
    Weights can be tuned.
    """
    return 0.5 * gfcc_energy + 0.3 * doppler_rms + 0.2 * doppler_max


# ============================================================
# DATA LOADING & SPLITTING
# ============================================================

def list_dataset_files(root: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Returns:
      files[user]["legitimate"] = [paths...]
      files[user]["attacker"]   = [paths...]
    """
    files = {}
    for user in os.listdir(root):
        user_path = os.path.join(root, user)
        if not os.path.isdir(user_path):
            continue

        legit_dir = os.path.join(user_path, "legitimate")
        attack_dir = os.path.join(user_path, "attacker")

        legit_files = []
        attack_files = []

        if os.path.isdir(legit_dir):
            for fname in sorted(os.listdir(legit_dir)):
                if fname.lower().endswith(".wav"):
                    legit_files.append(os.path.join(legit_dir, fname))

        if os.path.isdir(attack_dir):
            for fname in sorted(os.listdir(attack_dir)):
                if fname.lower().endswith(".wav"):
                    attack_files.append(os.path.join(attack_dir, fname))

        if legit_files or attack_files:
            files[user] = {
                "legitimate": legit_files,
                "attacker": attack_files
            }

    return files


def split_train_test(file_list: List[str], split_ratio: float) -> Tuple[List[str], List[str]]:
    """
    Random split of a list into train/test.
    If split_ratio >= 1, use full list for both train and test.
    """
    if len(file_list) == 0:
        return [], []

    if split_ratio >= 1.0:
        return file_list, file_list

    idx = np.arange(len(file_list))
    np.random.shuffle(idx)
    n_train = max(1, int(len(file_list) * split_ratio))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:] if n_train < len(file_list) else train_idx  # if all train, test = train

    train_files = [file_list[i] for i in train_idx]
    test_files = [file_list[i] for i in test_idx]

    return train_files, test_files


# ============================================================
# SCORING UTILITIES
# ============================================================

def compute_scores_for_file(user: str, path: str, label: int) -> Dict:
    """
    Compute GFCC + Doppler scores for a single file, with prints.
    label: 1 = legitimate, 0 = attacker
    """
    
    #Time
    start = time.time()
    
    gfcc = extract_gfcc_sequence(path)
    f_D, env = extract_doppler_features(path)

    gfcc_energy = gfcc.mean()
    doppler_rms = float(np.sqrt(np.mean(env ** 2)))
    doppler_max = float(np.max(np.abs(f_D)))
    score = float(compute_hybrid_score(gfcc_energy, doppler_rms, doppler_max))
    
    #Time
    end = time.time()
    feature_time = end - start

    label_str = "legitimate" if label == 1 else "attacker"
    print(f"[FEATURES] user={user:10s} label={label_str:11s}")
    print(f"           file={path}")
    print(f"           gfcc_energy={gfcc_energy:.6f}, doppler_rms={doppler_rms:.6f}, "
          f"doppler_max={doppler_max:.6f}, hybrid_score={score:.6f}")

    return {
        "user": user,
        "path": path,
        "label": label,
        "gfcc_energy": gfcc_energy,
        "doppler_rms": doppler_rms,
        "doppler_max": doppler_max,
        "score": score,
        #Time
        "feature_time": feature_time
    }


def find_best_threshold(scores: np.ndarray,
                        labels: np.ndarray,
                        n_steps: int = N_THRESHOLD_STEPS) -> Tuple[float, str]:
    """
    Find best threshold and direction (">" or "<") by minimizing ACER on given scores/labels.
    labels: 1 = legitimate, 0 = attacker
    """
    s_min, s_max = scores.min(), scores.max()
    if s_min == s_max:
        return float(s_min), ">"

    best_acer = float("inf")
    best_theta = None
    best_dir = None

    for theta in np.linspace(s_min, s_max, n_steps):
        for direction in (">", "<"):
            if direction == ">":
                preds = (scores > theta).astype(int)
            else:
                preds = (scores < theta).astype(int)

            TP = np.sum((preds == 1) & (labels == 1))
            FN = np.sum((preds == 0) & (labels == 1))
            TN = np.sum((preds == 0) & (labels == 0))
            FP = np.sum((preds == 1) & (labels == 0))

            FAR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
            FRR = FN / (TP + FN) if (TP + FN) > 0 else 0.0
            ACER = (FAR + FRR) / 2.0

            if ACER < best_acer:
                best_acer = ACER
                best_theta = theta
                best_dir = direction

    return float(best_theta), best_dir


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute EER by sweeping threshold on scores (higher score -> more likely legitimate).
    labels: 1 = legitimate, 0 = attacker
    """
    idx_sort = np.argsort(scores)
    scores_sorted = scores[idx_sort]
    labels_sorted = labels[idx_sort]

    # unique thresholds
    thresholds = np.unique(scores_sorted)
    eer_best = 1.0

    for theta in thresholds:
        preds = (scores >= theta).astype(int)
        TP = np.sum((preds == 1) & (labels == 1))
        FN = np.sum((preds == 0) & (labels == 1))
        TN = np.sum((preds == 0) & (labels == 0))
        FP = np.sum((preds == 1) & (labels == 0))

        FAR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
        FRR = FN / (TP + FN) if (TP + FN) > 0 else 0.0
        diff = abs(FAR - FRR)
        if diff < eer_best:
            eer_best = (FAR + FRR) / 2.0

    return eer_best


# ============================================================
# TRAIN + EVAL MODES
# ============================================================

def run_global_threshold_mode(files: Dict[str, Dict[str, List[str]]], split_ratio: float) -> None:
    """
    Global threshold across all users.
    """
    print("\n===== MODE: GLOBAL THRESHOLD =====")

    train_entries = []
    test_entries = []

    # ---------- SPLIT & FEATURE EXTRACTION ----------
    for user, paths in files.items():
        legit_files = paths.get("legitimate", [])
        attack_files = paths.get("attacker", [])

        legit_train, legit_test = split_train_test(legit_files, split_ratio)
        attack_train, attack_test = split_train_test(attack_files, split_ratio)

        print(f"\n[USER] {user}:")
        print(f"  Legitimate: train={len(legit_train)}, test={len(legit_test)}")
        print(f"  Attacker:   train={len(attack_train)}, test={len(attack_test)}\n")

        # Training legit
        for fpath in legit_train:
            entry = compute_scores_for_file(user, fpath, label=1)
            train_entries.append(entry)

        # Training attacker
        for fpath in attack_train:
            entry = compute_scores_for_file(user, fpath, label=0)
            train_entries.append(entry)

        # Testing legit
        for fpath in legit_test:
            entry = compute_scores_for_file(user, fpath, label=1)
            test_entries.append(entry)

        # Testing attacker
        for fpath in attack_test:
            entry = compute_scores_for_file(user, fpath, label=0)
            test_entries.append(entry)

    # ---------- TRAINING (GLOBAL THRESHOLD) ----------
    train_scores = np.array([e["score"] for e in train_entries])
    train_labels = np.array([e["label"] for e in train_entries])
    theta, direction = find_best_threshold(train_scores, train_labels)

    print("\n[TRAINING - GLOBAL]")
    print(f"  Optimal theta = {theta:.6f}, direction = '{direction}'")

    # ---------- TESTING ----------
    TP = TN = FP = FN = 0
    test_scores_raw = []
    test_labels = []

    print("\n[TESTING - GLOBAL]")

    for e in test_entries:
        score = e["score"]
        label = e["label"]
        user = e["user"]
        path = e["path"]

        # decision
        if direction == ">":
            pred = 1 if score > theta else 0
            # for EER scoring: higher score => more live-like
            s_aligned = score
        else:
            pred = 1 if score < theta else 0
            s_aligned = -score  # flip for EER alignment

        correct = (pred == label)
        result_str = "CORRECT" if correct else "WRONG"
        label_str = "legitimate" if label == 1 else "attacker"
        pred_str = "legitimate" if pred == 1 else "attacker"

        print(f"[TEST] user={user:10s} label={label_str:11s} pred={pred_str:11s} "
              f"score={score:.6f} => {result_str}")

        # accumulate counts
        if label == 1 and pred == 1:
            TP += 1
        elif label == 1 and pred == 0:
            FN += 1
        elif label == 0 and pred == 0:
            TN += 1
        elif label == 0 and pred == 1:
            FP += 1

        test_scores_raw.append(s_aligned)
        test_labels.append(label)

    # ---------- METRICS ----------
    test_scores_raw = np.array(test_scores_raw)
    test_labels = np.array(test_labels)

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total else 0.0
    FAR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    FRR = FN / (TP + FN) if (TP + FN) > 0 else 0.0
    EER = compute_eer(test_scores_raw, test_labels) if len(test_scores_raw) > 0 else 0.0

    print("\n===== GLOBAL THRESHOLD RESULTS =====")
    print(f"TP={TP}, FN={FN}, TN={TN}, FP={FP}")
    print(f"Accuracy = {accuracy:.4f}")
    print(f"FAR      = {FAR:.4f}")
    print(f"FRR      = {FRR:.4f}")
    print(f"EER      = {EER:.4f}")


def run_per_user_threshold_mode(files: Dict[str, Dict[str, List[str]]], split_ratio: float) -> None:
    """
    Per-user dynamic threshold using only that user's training data.
    """
    print("\n===== MODE: PER-USER DYNAMIC THRESHOLD =====")

    per_user_train = {}
    test_entries = []

    # ---------- SPLIT & FEATURE EXTRACTION ----------
    for user, paths in files.items():
        legit_files = paths.get("legitimate", [])
        attack_files = paths.get("attacker", [])

        legit_train, legit_test = split_train_test(legit_files, split_ratio)
        attack_train, attack_test = split_train_test(attack_files, split_ratio)

        print(f"\n[USER] {user}:")
        print(f"  Legitimate: train={len(legit_train)}, test={len(legit_test)}")
        print(f"  Attacker:   train={len(attack_train)}, test={len(attack_test)}\n")

        user_train_entries = []

        # Training legit
        for fpath in legit_train:
            entry = compute_scores_for_file(user, fpath, label=1)
            user_train_entries.append(entry)

        # Training attacker
        for fpath in attack_train:
            entry = compute_scores_for_file(user, fpath, label=0)
            user_train_entries.append(entry)

        per_user_train[user] = user_train_entries

        # Testing legit
        for fpath in legit_test:
            entry = compute_scores_for_file(user, fpath, label=1)
            test_entries.append(entry)

        # Testing attacker
        for fpath in attack_test:
            entry = compute_scores_for_file(user, fpath, label=0)
            test_entries.append(entry)

    # ---------- TRAINING (PER-USER THRESHOLDS) ----------
    per_user_thresholds = {}
    print("\n[TRAINING - PER USER]")

    for user, entries in per_user_train.items():
        if not entries:
            continue
        scores = np.array([e["score"] for e in entries])
        labels = np.array([e["label"] for e in entries])
        theta, direction = find_best_threshold(scores, labels)
        per_user_thresholds[user] = (theta, direction)
        print(f"  user={user:10s} theta={theta:.6f}, direction='{direction}'")

    # ---------- TESTING ----------
    TP = TN = FP = FN = 0
    test_scores_aligned = []
    test_labels = []
    
    test_times_feature = []
    test_times_decision = []

    
    print("\n[TEST EXECUTION]")
    

    print("\n[TESTING - PER USER]")

    for e in test_entries:
        
        # Time
        # (A) Record feature computation time
        test_times_feature.append(entry["feature_time"])
        # (B) Decision timing
        start_dec = time.time()
    
        user = e["user"]
        if user not in per_user_thresholds:
            # fallback: skip
            continue

        theta, direction = per_user_thresholds[user]
        score = e["score"]
        label = e["label"]

        if direction == ">":
            pred = 1 if score > theta else 0
            s_aligned = score
        else:
            pred = 1 if score < theta else 0
            s_aligned = -score
        
        # Time
        end_dec = time.time()
        test_times_decision.append(end_dec - start_dec)
        
        correct = (pred == label)
        result_str = "CORRECT" if correct else "WRONG"
        label_str = "legitimate" if label == 1 else "attacker"
        pred_str = "legitimate" if pred == 1 else "attacker"

        print(f"[TEST] user={user:10s} label={label_str:11s} pred={pred_str:11s} "
              f"score={score:.6f} (theta={theta:.6f}, dir={direction}) => {result_str}")

        if label == 1 and pred == 1:
            TP += 1
        elif label == 1 and pred == 0:
            FN += 1
        elif label == 0 and pred == 0:
            TN += 1
        elif label == 0 and pred == 1:
            FP += 1

        test_scores_aligned.append(s_aligned)
        test_labels.append(label)

    # ---------- METRICS ----------
    test_scores_aligned = np.array(test_scores_aligned)
    test_labels = np.array(test_labels)
    
    # Timer ends
    feature_avg = np.mean(test_times_feature)
    feature_total = np.sum(test_times_feature)

    decision_avg = np.mean(test_times_decision)
    decision_total = np.sum(test_times_decision)

    print("\n=== TIMING RESULTS ===")
    print(f"Test samples             : {len(test_times_feature)}")
    print(f"Total feature time (s)   : {feature_total:.6f}")
    print(f"Avg. feature time (s)    : {feature_avg:.6f}")
    print(f"Total decision time (s)  : {decision_total:.6f}")
    print(f"Avg. decision time (s)   : {decision_avg:.6f}")

    overall_avg = feature_avg + decision_avg
    print(f"\nAvg. total per-sample (s): {overall_avg:.6f}")

    total = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total else 0.0
    FAR = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    FRR = FN / (TP + FN) if (TP + FN) > 0 else 0.0
    #EER = compute_eer(test_scores_aligned, test_labels) if len(test_scores_aligned) > 0 else 0.0
    EER = (FAR + FRR) / 2

    print("\n===== PER-USER THRESHOLD RESULTS =====")
    print(f"TP={TP}, FN={FN}, TN={TN}, FP={FP}")
    print(f"Accuracy = {accuracy:.4f}")
    print(f"FAR      = {FAR:.4f}")
    print(f"FRR      = {FRR:.4f}")
    print(f"EER      = {EER:.4f}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n=== BlowLive Liveness Evaluation ===")
    print(f"Dataset root : {DATASET_ROOT}")
    #print(f"Train/Test split ratio: {SPLIT_RATIO}")
    print("\nModes:")
    print("  1. Global Threshold")
    print("  2. Per-User Dynamic Threshold")
    mode = input("Select mode [1/2]: ").strip()

    files = list_dataset_files(DATASET_ROOT)
    if not files:
        print("No users or .wav files found in dataset structure.")
        return

    if mode == "1":
        run_global_threshold_mode(files, SPLIT_RATIO)
    elif mode == "2":
        run_per_user_threshold_mode(files, SPLIT_RATIO)
    else:
        print("Invalid mode selection.")


if __name__ == "__main__":
    main()

