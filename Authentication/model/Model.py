# improved_multimodal_training.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# If you use facenet-pytorch, keep it installed. Otherwise swap with your FaceNet loader.
from facenet_pytorch import InceptionResnetV1
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
    
from torchvision import transforms

image_transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])