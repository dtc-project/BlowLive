import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def binarize_median(embedding):
    return (embedding > np.median(embedding)).astype(np.uint8)

import os
import pandas as pd
import numpy as np

def binarize_median(embedding):
    return (embedding > np.median(embedding)).astype(np.uint8)

def save_binary_embeddings(root_dir, suffix_map):
    """
    Convert embeddings to binary (median rule) and save them back with '_binary_' in the filename.
    """
    for subfolder in os.listdir(root_dir):
        subdir_path = os.path.join(root_dir, subfolder)
        if not os.path.isdir(subdir_path):
            continue

        for file in os.listdir(subdir_path):
            for key, suffix in suffix_map.items():
                if file.endswith(suffix):
                    file_path = os.path.join(subdir_path, file)
                    try:
                        df = pd.read_csv(file_path, header=None)
                        embedding = df.values.flatten()

                        # Binarize
                        binary_embedding = binarize_median(embedding)

                        # Build new filename
                        new_file = file.replace(suffix, f"_binary{suffix}")
                        new_path = os.path.join(subdir_path, new_file)

                        # Save
                        pd.DataFrame(binary_embedding).to_csv(new_path, header=False, index=False)
                        print(f"✅ Saved binary file: {new_path}")
                    except Exception as e:
                        print(f"❌ Failed {file_path}: {e}")

if __name__ == "__main__":
    root_dir = "BlowPrintData"
    suffix_map = {
        "new_image": "_new_image_emb.csv",
        "new_audio": "_new_audio_emb.csv",
        "new_combined": "_new_combined_emb.csv",
    }

    save_binary_embeddings(root_dir, suffix_map)

