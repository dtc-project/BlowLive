import pandas as pd
import numpy as np
from scipy.io.wavfile import write
import os

#Access the root folder 
base_dir = os.getcwd() + "/BlowPrintData"

# Wav Sample Rate
sample_rate = 48000 

#Process all found RawAudio Data in the directory
# Search and process images
for dirpath, _, filenames in os.walk(base_dir):
    for filename in filenames:
        if "_RawAudio_" in filename and ".wav" not in filename:
            file_path = os.path.join(dirpath, filename)
            # print(f"Processing {file_path}...")

            # Save embedding to CSV
            wav_filename = filename.replace(".csv", ".wav")
            wav_path = os.path.join(dirpath, wav_filename)

            if os.path.exists(wav_path):
                print(f"WAV file already exists: {wav_path}, skipping...")
                continue
            
            # Read CSV
            try:
                df = pd.read_csv(file_path)
                all_samples = []

                # Parse semicolon-separated values
                for row in df[' Raw Audio Data']:
                    samples = row.replace('"', '').split(';')
                    all_samples.extend([float(val) for val in samples if val.strip() != ''])

                # Convert to int16 waveform
                signal = np.array(all_samples)
                signal = signal - np.mean(signal)
                signal = signal / np.max(np.abs(signal))
                signal = (signal * 32767).astype(np.int16)

                # Write to WAV
                write(wav_path, sample_rate, signal)
                print(f"Saved WAV to: {wav_path}")
            except Exception as e:
                print(f"Failed to process {file_path}: {e}")
