import os
import random
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.nn.functional import cosine_similarity
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_curve
import matplotlib.pyplot as plt
from model.Model import *
import pandas as pd  # <- You missed this

def load_gfcc_csv(csv_path, target_len=265):
        gfcc_feat = np.loadtxt(csv_path, delimiter=',')
        if gfcc_feat.shape[0] > target_len:
            gfcc_feat = gfcc_feat[:target_len]
        else:
            pad_len = target_len - gfcc_feat.shape[0]
            gfcc_feat = np.pad(gfcc_feat, ((0, pad_len), (0, 0)))
        return torch.tensor(gfcc_feat.T, dtype=torch.float32)  # shape: (26, 260)

def save_embedding(embedding: torch.Tensor, path: str):
    df = pd.DataFrame(embedding.cpu().numpy().squeeze())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, header=False)

def process_and_save_embeddings(root_dir):
    for user in os.listdir(root_dir):
        user_dir = os.path.join(root_dir, user)
        if not os.path.isdir(user_dir):
            continue


        jpg_files = glob.glob(os.path.join(user_dir, "*.jpg"))
        csv_files = glob.glob(os.path.join(user_dir, "*_gfcc.csv"))

        if not jpg_files or not csv_files:
            continue

        image_path = jpg_files[0]
        gfcc_path = csv_files[0]

        try:
            image = Image.open(image_path).convert("RGB")
            image = image_transform(image).unsqueeze(0).to(device)

            gfcc = load_gfcc_csv(gfcc_path, target_len=gfcc_len)
            gfcc = gfcc.unsqueeze(0).to(device)

            with torch.no_grad():
                image_emb, audio_emb, combined_emb,norm_img,norm_audio = model(image, gfcc)
            
            save_embedding(image_emb, os.path.join(user_dir, "new_image_emb.csv"))
            save_embedding(audio_emb, os.path.join(user_dir, "new_audio_emb.csv"))
            save_embedding(combined_emb, os.path.join(user_dir, "new_combined_emb.csv"))
            # save_embedding(norm_img, os.path.join(user_dir, "new_normimage_emb.csv"))
            # save_embedding(norm_audio, os.path.join(user_dir, "new_normaudio_emb.csv"))

        except Exception as e:
            print(f"Error processing {user_dir}: {e}")


image_transform = image_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Settings ===
root_dir = "BlowPrintData"
output_dim = 128
gfcc_len = 265

model = ImageGFCCContrastive(output_dim=output_dim).to(device)
model.load_state_dict(torch.load("checkpoints/GFCC_Facenet.pth"))
model.eval()
# --- Main ---
def main():

    print(1)
    # Call the function
    process_and_save_embeddings(root_dir)

main()