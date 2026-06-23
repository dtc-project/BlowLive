# improved_multimodal_training.py
import os
import glob
import random
from collections import defaultdict, Counter
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import accuracy_score, roc_curve

# If you use facenet-pytorch, keep it installed. Otherwise swap with your FaceNet loader.
from facenet_pytorch import InceptionResnetV1

# ---------------------------
# Utilities
# ---------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_eer_from_scores(y_score, y_true):
    # y_true: list of 0/1, y_score: floats
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[idx]
    thr = thresholds[idx]
    return eer, thr

# ---------------------------
# Models
# ---------------------------

class ImageNetFace(nn.Module):
    """FaceNet backbone with small projection head. Optionally freeze backbone."""
    def __init__(self, output_dim=128, freeze_backbone=True):
        super().__init__()
        self.backbone = InceptionResnetV1(pretrained='vggface2', classify=False)
        self.backbone.eval() if freeze_backbone else self.backbone.train()
        # don't require grads for backbone if freezing
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        self.proj = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        # backbone may be frozen; still call it normally
        emb = self.backbone(x)                  # (B,512)
        emb = self.proj(emb)                    # (B,output_dim)
        emb = F.normalize(emb, p=2, dim=1)
        return emb

class GFCC_CNN(nn.Module):
    def __init__(self, input_dim=26, output_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        # x: (B, C=26, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.mean(x, dim=-1)   # temporal pooling -> (B, 256)
        x = F.relu(self.fc1(x))
        x = F.normalize(self.fc2(x), p=2, dim=1)
        return x

class ImageGFCCContrastive(nn.Module):
    def __init__(self, output_dim=128, freeze_face_backbone=True):
        super().__init__()
        self.image_net = ImageNetFace(output_dim=output_dim, freeze_backbone=freeze_face_backbone)
        self.gfcc_net = GFCC_CNN(input_dim=26, output_dim=output_dim)

        # small projector heads (trainable) — used for contrastive alignment
        self.image_project = nn.Sequential(nn.Linear(output_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim))
        self.gfcc_project = nn.Sequential(nn.Linear(output_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim))

    def forward(self, image, gfcc):
        # image: (B,3,160,160), gfcc: (B, 26, T)
        img_emb = self.image_net(image)        # (B, output_dim)
        gfcc_emb = self.gfcc_net(gfcc)         # (B, output_dim)

        img_proj = self.image_project(img_emb)
        gfcc_proj = self.gfcc_project(gfcc_emb)

        # Combined representation (simple sum + normalize)
        fused = F.normalize(img_proj + gfcc_proj, p=2, dim=1)

        # return normalized branch embeddings and the fused projection
        return img_emb, gfcc_emb, fused, F.normalize(img_proj, p=2, dim=1), F.normalize(gfcc_proj, p=2, dim=1)

# ---------------------------
# Losses
# ---------------------------

class TripletLoss(nn.Module):
    def __init__(self, margin=0.7):
        super().__init__()
        self.margin = margin
    def forward(self, anchor, positive, negative):
        pd = F.pairwise_distance(anchor, positive)
        nd = F.pairwise_distance(anchor, negative)
        return torch.mean(F.relu(pd - nd + self.margin))

class NTXentLoss(nn.Module):
    """Normalized temperature-scaled cross entropy (InfoNCE) for cross-modal alignment.
       We will compute similarity between matched pairs (image_proj, gfcc_proj) in a batch.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.cos = nn.CosineSimilarity(dim=2)

    def forward(self, a_proj, b_proj):
        # a_proj, b_proj: (B, D) and should be normalized beforehand
        # We'll build a 2B x 2B similarity matrix and use cross-entropy
        a = a_proj
        b = b_proj
        batch_size = a.shape[0]

        z = torch.cat([a, b], dim=0)  # (2B, D)
        sim = torch.matmul(z, z.t()) / self.temperature  # (2B,2B) (cosine if normalized)
        # mask out self-similarity
        mask = (~torch.eye(2 * batch_size, 2 * batch_size, dtype=torch.bool)).to(sim.device)
        sim_masked = sim.masked_select(mask).view(2 * batch_size, -1)

        # positives: for index i in [0,B) positive is i+B ; for i in [B,2B) positive is i-B
        positives = torch.cat([torch.diag(sim, batch_size), torch.diag(sim, -batch_size)], dim=0) / self.temperature
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=sim.device)
        logits = torch.cat([positives.unsqueeze(1), sim_masked], dim=1)
        loss = F.cross_entropy(logits, labels)
        return loss

# ---------------------------
# Dataset
# ---------------------------

class GFCCImageTripletDataset(Dataset):
    """
    Directory-level dataset: each element in dir_list is a folder containing
    *_gfcc260-26.csv and corresponding .jpg with same prefix (username_session...).
    preloads GFCC (padded to fixed length) and images (transformed) into memory.
    """
    def __init__(self, dir_list, transform_image=None, gfcc_fixed_len=265):
        self.transform_image = transform_image
        self.gfcc_fixed_len = gfcc_fixed_len
        self.user_data = defaultdict(list)
        # preloaded_data[user] = list of (gfcc_tensor, img_tensor)
        self.preloaded_data = defaultdict(list)

        for session_path in dir_list:
            csvs = glob.glob(os.path.join(session_path, "*_gfcc260-26.csv"))
            jpgs = glob.glob(os.path.join(session_path, "*.jpg"))
            jpg_map = {os.path.basename(f).split("_")[0]: f for f in jpgs}
            for csv in csvs:
                prefix = os.path.basename(csv).split("_")[0]
                jpg = jpg_map.get(prefix)
                if not jpg:
                    continue
                gfcc = np.loadtxt(csv, delimiter=",")
                # pad or crop temporal dim
                if gfcc_fixed_len:
                    if gfcc.shape[0] > gfcc_fixed_len:
                        gfcc = gfcc[:gfcc_fixed_len, :]
                    else:
                        pad_len = gfcc_fixed_len - gfcc.shape[0]
                        gfcc = np.pad(gfcc, ((0, pad_len), (0, 0)))
                # transpose to (C=26, T)
                gfcc_t = torch.tensor(gfcc.T, dtype=torch.float32)

                img = Image.open(jpg).convert("RGB")
                if self.transform_image:
                    img_t = self.transform_image(img)
                else:
                    img_t = transforms.ToTensor()(img)

                # Use username as prefix for grouping
                username = prefix
                self.user_data[username].append((csv, jpg))
                self.preloaded_data[username].append((gfcc_t, img_t))

        self.user_list = list(self.preloaded_data.keys())
        self.all_samples = [(u, i) for u in self.user_list for i in range(len(self.preloaded_data[u]))]

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        u, idx_within = self.all_samples[idx]
        # positive sample
        pos_indices = list(range(len(self.preloaded_data[u])))
        pos_indices.remove(idx_within)
        if len(pos_indices) == 0:
            pos_idx = idx_within
        else:
            pos_idx = random.choice(pos_indices)

        # negative (from different user)
        neg_user = random.choice([x for x in self.user_list if x != u])
        neg_idx = random.randint(0, len(self.preloaded_data[neg_user]) - 1)

        a_g, a_i = self.preloaded_data[u][idx_within]
        p_g, p_i = self.preloaded_data[u][pos_idx]
        n_g, n_i = self.preloaded_data[neg_user][neg_idx]

        return (a_i, a_g), (p_i, p_g), (n_i, n_g), u

# ---------------------------
# Training and evaluation helpers
# ---------------------------

def train_epoch(model, loader, loss_triplet_branch, loss_triplet_fused, loss_ntx, optimizer, device, grad_clip=1.0):
    model.train()
    running_loss = 0.0
    for a, p, n, _ in tqdm(loader, desc="Train", leave=False):
        (a_img, a_gfcc) = a
        (p_img, p_gfcc) = p
        (n_img, n_gfcc) = n

        a_img = a_img.to(device); p_img = p_img.to(device); n_img = n_img.to(device)
        a_gfcc = a_gfcc.to(device); p_gfcc = p_gfcc.to(device); n_gfcc = n_gfcc.to(device)

        optimizer.zero_grad()

        a_img_emb, a_g_emb, a_fused, a_img_proj, a_g_proj = model(a_img, a_gfcc)
        p_img_emb, p_g_emb, p_fused, p_img_proj, p_g_proj = model(p_img, p_gfcc)
        n_img_emb, n_g_emb, n_fused, n_img_proj, n_g_proj = model(n_img, n_gfcc)

        # Branch-specific triplet losses (keep audio branch strong standalone)
        loss_a_branch = loss_triplet_branch(a_g_emb, p_g_emb, n_g_emb)
        loss_i_branch = loss_triplet_branch(a_img_emb, p_img_emb, n_img_emb)

        # Fused triplet: anchor fused vs positive fused vs negative fused
        loss_fused = loss_triplet_fused(a_fused, p_fused, n_fused)

        # Cross-modal alignment (NTXent) between projections in the batch (image_proj vs gfcc_proj)
        # We'll compute it using anchor/positive pairs and include all three anchors (a and p are same user)
        # Build small batches for NT-Xent: stack (a_img_proj, p_img_proj) vs (a_g_proj, p_g_proj)
        img_proj_batch = torch.cat([a_img_proj, p_img_proj], dim=0)
        gfcc_proj_batch = torch.cat([a_g_proj, p_g_proj], dim=0)
        loss_align = loss_ntx(img_proj_batch, gfcc_proj_batch)

        # total loss with weighting (tweak lambdas as needed)
        total_loss = 1.0 * loss_a_branch + 1.0 * loss_i_branch + 1.0 * loss_fused + 1.0 * loss_align

        #without NTXent
        # total_loss = 1.0 * loss_a_branch + 0.5 * loss_i_branch + 1.0 * loss_fused
        

        total_loss.backward()
        # optional gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += total_loss.item()
    return running_loss / len(loader)

def collect_embeddings(dataset, model, device, image_only=False, audio_only=False):
    model.eval()
    embs = []
    labels = []
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            (img, gfcc), _, _, label = sample
            img = img.unsqueeze(0).to(device)
            gfcc = gfcc.unsqueeze(0).to(device)
            img_emb, g_emb, fused, img_proj, gproj = model(img, gfcc)
            if image_only:
                emb = img_emb
            elif audio_only:
                emb = g_emb
            else:
                emb = fused
            embs.append(emb.squeeze(0).cpu())
            labels.append(label)
    return torch.stack(embs), labels

def build_reference_from_trainset(trainset, model, device, image_only=False, audio_only=False):
    model.eval()
    reference = {}
    with torch.no_grad():
        for user in trainset.user_list:
            samples = trainset.preloaded_data[user]  # list of (gfcc, img)
            embs = []
            for gfcc, img in samples:
                img = img.unsqueeze(0).to(device)
                gfcc = gfcc.unsqueeze(0).to(device)
                img_emb, g_emb, fused, _, _ = model(img, gfcc)
                if image_only:
                    emb = img_emb
                elif audio_only:
                    emb = g_emb
                else:
                    emb = fused
                embs.append(emb.squeeze(0).cpu())
            if len(embs) > 0:
                reference[user] = torch.stack(embs).mean(dim=0)
    return reference

def predict_sample(sample, model, reference, device, image_only=False, audio_only=False):
    (img, gfcc), _, _, label = sample
    img = img.unsqueeze(0).to(device)
    gfcc = gfcc.unsqueeze(0).to(device)
    with torch.no_grad():
        img_emb, g_emb, fused, _, _ = model(img, gfcc)
        if image_only:
            emb = img_emb
        elif audio_only:
            emb = g_emb
        else:
            emb = fused
    similarities = {user: F.cosine_similarity(emb.cpu(), ref.unsqueeze(0)).item() for user, ref in reference.items()}
    pred_user = max(similarities, key=similarities.get)
    return pred_user, label, similarities[pred_user]

def test(model, train_ds, test_ds, device, image_only=False, audio_only=False, topk=3):
    # Build reference (or pre-compute train embeddings)
    reference = build_reference_from_trainset(train_ds, model, device, image_only=image_only, audio_only=audio_only)
    # For K-NN classification we'll use all train embeddings stacked
    train_embs, train_labels = collect_embeddings(train_ds, model, device, image_only=image_only, audio_only=audio_only)
    model.eval()

    all_preds, all_labels = [], []
    for i in range(len(test_ds)):
        sample = test_ds[i]
        (img, gfcc), _, _, true_label = sample
        img = img.unsqueeze(0).to(device); gfcc = gfcc.unsqueeze(0).to(device)
        with torch.no_grad():
            img_emb, g_emb, fused, _, _ = model(img, gfcc)
            if image_only:
                emb = img_emb
            elif audio_only:
                emb = g_emb
            else:
                emb = fused
        sims = F.cosine_similarity(emb.cpu(), train_embs).squeeze(0)
        topk_idx = torch.topk(sims, k=topk).indices.tolist()
        topk_labels = [train_labels[idx] for idx in topk_idx]
        pred_label = Counter(topk_labels).most_common(1)[0][0]
        all_preds.append(pred_label)
        all_labels.append(true_label)

    # classification accuracy
    label_to_idx = {name: idx for idx, name in enumerate(sorted(set(all_labels)))}
    acc = accuracy_score([label_to_idx[l] for l in all_labels], [label_to_idx[p] for p in all_preds])

    all_preds, all_labels = [], []
    for i in range(len(test_ds)):
        sample = test_ds[i]
        pred, label, _ = predict_sample(sample, model, reference, device, image_only, audio_only)
        all_preds.append(pred)
        all_labels.append(label)
    label_to_idx = {name: idx for idx, name in enumerate(set(all_labels))}
    accuracy = accuracy_score([label_to_idx[l] for l in all_labels], [label_to_idx[p] for p in all_preds])
    


    # authentication (EER): compute score distributions per user
    eer_scores = []
    for user in reference.keys():
        y_true = []
        y_score = []
        for i in range(len(test_ds)):
            sample = test_ds[i]
            (img, gfcc), _, _, label = sample
            img = img.unsqueeze(0).to(device); gfcc = gfcc.unsqueeze(0).to(device)
            with torch.no_grad():
                img_emb, g_emb, fused, _, _ = model(img, gfcc)
                if image_only:
                    emb = img_emb
                elif audio_only:
                    emb = g_emb
                else:
                    emb = fused
            user_embs = []
            # gather train embeddings for this user from train_ds
            # we can compute similarity against reference user mean to be efficient
            ref_vec = reference[user]
            sim = F.cosine_similarity(emb.cpu(), ref_vec.unsqueeze(0)).item()
            y_score.append(sim)
            y_true.append(1 if label == user else 0)
        eer, _ = compute_eer_from_scores(y_score, y_true)
        eer_scores.append(eer)
    avg_eer = float(np.mean(eer_scores)) if len(eer_scores) > 0 else float('nan')

    return acc, accuracy, avg_eer

def compute_authentication_metrics(scores, labels):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer, thresholds[np.nanargmin(np.absolute((fnr - fpr)))]


def test2(model, trainset, testset, device, image_only=False, audio_only=False):
    model.eval()
    print("\nCollecting train embeddings...")
    reference = build_reference_from_trainset(trainset, model, device, image_only=image_only, audio_only=audio_only)
    train_embs, train_labels = collect_embeddings(trainset, model, device, image_only, audio_only)

    all_preds, all_labels = [], []
    for i in tqdm(range(len(testset)), desc="Classifying"):
        sample = testset[i]
        img, gfcc = sample[0]
        true_label = sample[3]

        img = img.unsqueeze(0).to(device)
        gfcc = gfcc.unsqueeze(0).to(device)

        with torch.no_grad():
            if image_only:
                emb, _, _, _, _ = model(img, gfcc)
            elif audio_only:
                _, emb, _, _, _ = model(img, gfcc)
            else:
                _, _, emb, _, _ = model(img, gfcc)

        # Compute cosine similarity to all train embeddings
        sims = F.cosine_similarity(emb, train_embs).squeeze(0)  # shape: (num_train,)
        topk_idx = torch.topk(sims, k=3).indices.tolist()
        topk_labels = [train_labels[idx] for idx in topk_idx]
        pred_label = Counter(topk_labels).most_common(1)[0][0]

        all_preds.append(pred_label)
        all_labels.append(true_label)

    label_to_idx = {name: idx for idx, name in enumerate(set(all_labels))}
    accuracy = accuracy_score([label_to_idx[l] for l in all_labels], [label_to_idx[p] for p in all_preds])
    print(f"\nClassification Accuracy (3-NN): {accuracy:.4f}")

    all_preds, all_labels = [], []
    for i in tqdm(range(len(testset)), desc="Classifying"):
        sample = testset[i]
        pred, label ,_= predict_sample(sample, model, reference, device, image_only, audio_only)
        all_preds.append(pred)
        all_labels.append(label)
    label_to_idx = {name: idx for idx, name in enumerate(set(all_labels))}
    accuracy_r = accuracy_score([label_to_idx[l] for l in all_labels], [label_to_idx[p] for p in all_preds])
    print(f"\nClassification Reference Accuracy: {accuracy_r:.4f}")


    # --- Authentication ---
    eer_scores = []
    for user in tqdm(set(train_labels), desc="Authenticating"):
        y_true, y_score = [], []
        for i in range(len(testset)):
            sample = testset[i]
            img, gfcc = sample[0]
            label = sample[3]

            img = img.unsqueeze(0).to(device)
            gfcc = gfcc.unsqueeze(0).to(device)

            with torch.no_grad():
                if image_only:
                    emb, _, _, _, _ = model(img, gfcc)
                elif audio_only:
                    _, emb, _, _, _ = model(img, gfcc)
                else:
                    _, _, emb, _, _ = model(img, gfcc)

            # Get similarity to train samples belonging to the current user
            user_embs = train_embs[[i for i, lbl in enumerate(train_labels) if lbl == user]]
            sims = F.cosine_similarity(emb, user_embs)
            max_sim = torch.max(sims).item()  # top-1 similarity to that user

            y_score.append(max_sim)
            y_true.append(1 if label == user else 0)

        eer, _ = compute_authentication_metrics(y_score, y_true)
        eer_scores.append(eer)

    avg_eer = sum(eer_scores) / len(eer_scores)
    print(f"Average EER: {avg_eer:.4f}")

    return accuracy,accuracy_r, avg_eer


# ---------------------------
# Splitting helper
# ---------------------------
def split_train_test_by_username(root, ratio=0.7, min_sessions=10):
    us = defaultdict(list)
    for d in os.listdir(root):
        p = os.path.join(root, d)
        if os.path.isdir(p):
            u = d.split("_")[0]
            us[u].append(p)
    tr, te = [], []
    for u, s in us.items():
        if len(s) < min_sessions:
            continue
        random.shuffle(s)
        split_idx = int(ratio * len(s))
        tr += s[:split_idx]
        te += s[split_idx:]
    return tr, te

# ---------------------------
# Main training script
# ---------------------------
def main():
    root_dir = "BlowPrintData"   # adjust to your data path
    epochs = 50
    batch_size = 32
    lr = 1e-4
    gfcc_len = 265

    train_dirs, test_dirs = split_train_test_by_username(root_dir, ratio=0.7, min_sessions=10)

    tf = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
    ])

    train_ds = GFCCImageTripletDataset(train_dirs, transform_image=tf, gfcc_fixed_len=gfcc_len)
    test_ds = GFCCImageTripletDataset(test_dirs, transform_image=tf, gfcc_fixed_len=gfcc_len)

    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model = ImageGFCCContrastive(output_dim=128, freeze_face_backbone=True).to(device)

    # losses
    triplet_branch = TripletLoss(margin=0.8)
    triplet_fused = TripletLoss(margin=0.8)
    ntx = NTXentLoss(temperature=0.1)

    # optimizer: train gfcc_net, projectors, and image proj/FC (backbone may be frozen)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)

    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")
    for ep in range(epochs):
        loss = train_epoch(model, loader, triplet_branch, triplet_fused, ntx, optimizer, device)
        scheduler.step(loss)
        print(f"Epoch {ep+1}/{epochs}  avg_loss: {loss:.4f}")

    # final save
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/GFCC_Facenet_NTXEnt.pth") #Model Saved
    print("Saved final model.")

    model.load_state_dict(torch.load("checkpoints/GFCC_Facenet_NTXEnt.pth"))



    # final evaluation
    print("Final full eval...")
    acc_i, accr_i, eer_i = test2(model, train_ds, test_ds, device, image_only=True)
    acc_a, accr_a, eer_a = test2(model, train_ds, test_ds, device, audio_only=True)
    acc_f, accr_f, eer_f = test2(model, train_ds, test_ds, device)
    print("Results:")
    print("Image-only:", acc_i, accr_i, eer_i)
    print("Audio-only:", acc_a, accr_a, eer_a)
    print("Fused:", acc_f, accr_f, eer_f)

if __name__ == "__main__":
    main()
