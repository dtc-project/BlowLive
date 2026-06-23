import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Model import *
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ------------------- Data Utils -------------------

def get_csv_files(base_dir):
    files = []
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if "RawAudio" in file and file.endswith("_gfcc260-26.csv"):
                    files.append(os.path.join(folder_path, file))
    return files

def load_gfcc_and_user(csv_file):
    username = os.path.basename(csv_file).split("_")[0]
    features = np.loadtxt(csv_file, delimiter=",")
    return {"username": username, "features": features}

def normalize_features(data_list, mean, std):
    features = [(data["features"] - mean) / std for data in data_list]
    usernames = [data["username"] for data in data_list]
    return features, usernames

def pad_features(feature_series):
    max_len = max(f.shape[0] for f in feature_series)
    return [
        F.pad(torch.tensor(f.T, dtype=torch.float32), (0, max_len - f.shape[0]))
        for f in feature_series
    ]

def binarize_median(embedding):
    return (embedding > np.median(embedding)).astype(np.uint8)

# ------------------- Load Data -------------------

base_dir = os.path.join(os.getcwd(), "BlowPrintData")
csv_files = get_csv_files(base_dir)
all_data = [load_gfcc_and_user(f) for f in csv_files]

# Compute normalization stats
all_features = np.vstack([data["features"] for data in all_data])
mean, std = np.mean(all_features, axis=0), np.std(all_features, axis=0)

normed_feats, usernames = normalize_features(all_data, mean, std)
tensors = pad_features(normed_feats)

# ------------------- Load Model -------------------

model = GFCC_CNN().to(device)
model.load_state_dict(torch.load("gfcc_triplet_model_split.pth", map_location=device))
model.eval()

# ------------------- Extract & Save Embeddings to CSV -------------------

saved_users = set()

for tensor, username, csv_path in zip(tensors, usernames, csv_files):
    if username in saved_users:
        continue  # Avoid overwriting if already saved for that user

    with torch.no_grad():
        emb = model(tensor.unsqueeze(0).to(device)).cpu().numpy().flatten()
        bin_emb = binarize_median(emb)

    # Get user folder and create filenames
    user_dir = os.path.dirname(csv_path)
    emb_file = os.path.join(user_dir, f"{username}_embedding_split.csv")
    bin_file = os.path.join(user_dir, f"{username}_binary_split.csv")

    # Save as CSV
    np.savetxt(emb_file, emb, delimiter=",", fmt="%.6f")
    np.savetxt(bin_file, bin_emb, delimiter=",", fmt="%d")

    # saved_users.add(username)

print(f"✅ Saved embeddings and binaries for {len(saved_users)} users.")
