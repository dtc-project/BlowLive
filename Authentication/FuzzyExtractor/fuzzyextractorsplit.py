import os
import numpy as np
from glob import glob
from scipy.spatial.distance import hamming
from collections import Counter, defaultdict
import re
import bitarray
from fuzzy_reusable import *

# ------------------- Config -------------------

base_dir = os.path.join(os.getcwd(), "BlowLive_Dataset")

def get_binary_files(base_dir):
    all_files = glob(os.path.join(base_dir, "**", "*_audio.csv"), recursive=True)
    print(all_files[:10])

    files_A = [f for f in all_files if "_sit_" in f]
    files_B = [f for f in all_files if "_stand_" in f]


    return files_A, files_B


def read_binary_embeddings(binary_files):
    embeddings = []
    usernames = []
    for filepath in binary_files:
        binary_vec = np.loadtxt(filepath, delimiter=",", dtype=np.uint8)
        username = os.path.basename(filepath).split("_")[0]  # e.g. "user1"
        embeddings.append(binary_vec)
        usernames.append(username)
    return usernames, np.array(embeddings)


# ------------------- Main -------------------

if __name__ == "__main__":
    files_A, files_B = get_binary_files(base_dir)

    usernames_A, emb_A = read_binary_embeddings(files_A)
    usernames_B, emb_B = read_binary_embeddings(files_B)

    print(f"✅ Loaded {len(emb_A)} from A-folders, {len(emb_B)} from B-folders")
    print(f"Example A shape: {emb_A[0].shape}, Example B shape: {emb_B[0].shape}")

     # --- Sort by username ---
    combined = sorted(zip(usernames_A, emb_A), key=lambda x: x[0])
    usernames, binary_embeddings_sit = zip(*combined)
    binary_embeddings_sit = np.array(binary_embeddings_sit)

    # Compute Hamming distance matrix (raw integer, not normalized)
    num_samples = len(binary_embeddings_sit)
    hamming_matrix = np.zeros((num_samples, num_samples), dtype=int)

    for i in range(num_samples):
        for j in range(num_samples):
            hamming_matrix[i, j] = np.sum(binary_embeddings_sit[i] != binary_embeddings_sit[j])
    np.savetxt("sit_hamming_matrix_new_audio.csv", hamming_matrix, fmt="%d", delimiter=",")


    # --- Sort by username ---
    combined = sorted(zip(usernames_B, emb_B), key=lambda x: x[0])
    usernames, binary_embeddings_stand = zip(*combined)
    binary_embeddings_stand = np.array(binary_embeddings_stand)

    # Compute Hamming distance matrix (raw integer, not normalized)
    num_samples = len(binary_embeddings_stand)
    hamming_matrix = np.zeros((num_samples, num_samples), dtype=int)

    for i in range(num_samples):
        for j in range(num_samples):
            hamming_matrix[i, j] = np.sum(binary_embeddings_stand[i] != binary_embeddings_stand[j])
    np.savetxt("stand_hamming_matrix_new_audio.csv", hamming_matrix, fmt="%d", delimiter=",")




