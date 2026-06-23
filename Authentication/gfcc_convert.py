import numpy as np
import os
from gfcc import *

#Set to the Dataset Directory
base_dir = os.getcwd() + "/BlowPrintData"
sample_rate = 48000  # 48kHz for your blow audio

# Process each WAV file
for dirpath, _, filenames in os.walk(base_dir):
    for filename in filenames:
        if filename.endswith(".wav"):
            wav_path = os.path.join(dirpath, filename)
            gfcc_path = wav_path.replace(".wav", "_gfcc260-26.csv") # set name accordingly

            if os.path.exists(gfcc_path):
                print(f"GFCC already exists: {gfcc_path}, skipping...")
                continue

            try:
                gfcc_feats = compute_gfcc(wav_path) #Compute GFCC 
                save_gfcc_csv(gfcc_feats,gfcc_path) #Saved the GFCC into csv
                print(f"Saved GFCC: {gfcc_path}")

            except Exception as e:
                print(f"Failed to process {wav_path}: {e}")