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

# ------------------- Read Binary Embeddings -------------------

def get_binary_files(base_dir):
    all_files = glob(os.path.join(base_dir, "**", "*_audio.csv"), recursive=True)
    return [f for f in all_files if f.endswith("_audio.csv")]


def read_binary_embeddings(binary_files):
    embeddings = []
    usernames = []
    for filepath in binary_files:
        binary_vec = np.loadtxt(filepath, delimiter=",", dtype=np.uint8)
        username = os.path.basename(filepath).split("_")[0]
        embeddings.append(binary_vec)
        usernames.append(username)
    return usernames, np.array(embeddings)

if __name__ == "__main__":
    binary_files = get_binary_files(base_dir)
    usernames, binary_embeddings = read_binary_embeddings(binary_files)

    print(f"✅ Loaded {len(binary_embeddings)} binary vectors.")
    print(f"Vector shape: {binary_embeddings[0].shape}")

    # # --- Sort by username ---
    combined = sorted(zip(usernames, binary_embeddings), key=lambda x: x[0])
    usernames, binary_embeddings = zip(*combined)
    binary_embeddings = np.array(binary_embeddings)

    # Compute Hamming distance matrix (raw integer, not normalized)
    num_samples = len(binary_embeddings)
    hamming_matrix = np.zeros((num_samples, num_samples), dtype=int)

    for i in range(num_samples):
        for j in range(num_samples):
            hamming_matrix[i, j] = np.sum(binary_embeddings[i] != binary_embeddings[j])
    np.savetxt("hamming_matrix_new_audio.csv", hamming_matrix, fmt="%d", delimiter=",")
    print("✅ Saved Hamming distance matrix to hamming_matrix.csv")


    #Some Testing of Revocation using Fuzzy Extractor
    A = binary_embeddings[1]  # shape (128,)
    B = binary_embeddings[2]
    C = binary_embeddings[21]
    D = binary_embeddings[3]


    A_bytes = np.packbits(A).tobytes()
    B_bytes = np.packbits(B).tobytes()
    C_bytes = np.packbits(C).tobytes()
    D_bytes = np.packbits(D).tobytes()



    key, helper, salt = fuzzy_generate_reusable(A_bytes)
    rep_key = fuzzy_reproduce_reusable(B_bytes, helper,salt)

    # Compute Hamming distance in bits
    hamming_distance = np.sum(A != B)
    print(f"Hamming distance A-B (bits): {hamming_distance}")

    print("Original key:", key.hex())
    if(rep_key != None):
        print("Reproduced key:", rep_key.hex())
        print("✅ Match:", key == rep_key)
        
    else:
        print("❌ No Key Reproduced!")

    key,helper,salt = fuzzy_revocation(C_bytes)
    rep_key = fuzzy_reproduce_reusable(B_bytes, helper,salt)
    # Compute Hamming distance in bits
    hamming_distance = np.sum(C != B)
    print("======================================\nRevoke Using Biometrics (C) - Different user to A and B")
    print(f"Hamming distance B-C(bits): {hamming_distance}")

    print("Original key:", key.hex())
    if(rep_key != None):

        print("Reproduced key:", rep_key.hex())
        print("✅ Match:", key == rep_key)
    else:
        print("❌ No Key Reproduced!")

    key,helper,salt = fuzzy_revocation(D_bytes)
    rep_key = fuzzy_reproduce_reusable(B_bytes, helper,salt)
    # Compute Hamming distance in bits
    hamming_distance = np.sum(D != B)
    print("======================================\nRevoke Using Biometrics (D) - Same user to A and B")
    print(f"Hamming distance B-D(bits): {hamming_distance}")

    print("Original key:", key.hex())
    if(rep_key != None):

        print("Reproduced key:", rep_key.hex())
        print("✅ Match:", key == rep_key)
    else:
        print("❌ No Key Reproduced!")
    