# BlowLive — Authentication Module

## Directory Guide

**`BlowLive-Dataset_Without_Doppler_Shift/`** — Dataset of 50 participants, each with 10 sessions (5 sit + 5 stand), 4 CSV files per session:

```
BlowLive-Dataset_Without_Doppler_Shift/
├── Participant 1
│   ├── session_1_sit_audio.csv
│   ├── session_1_sit_img.csv
│   ├── session_1_sit_gfcc.csv
│   ├── session_1_sit_combined.csv
│   ├── ...
│   ├── session_6_stand_audio.csv
│   ├── session_6_stand_img.csv
│   ├── session_6_stand_gfcc.csv
│   └── session_6_stand_combined.csv
├── Participant 2
│   └── ...
└── Participant 50
    └── ...
```

**`gfcc.py`** — GFCC feature extraction model (interval, deltas, and other settings)<br>
**`gfcc_convert.py`** — Converts `.wav` files to GFCC `.csv` format<br>
**`DTW_Calculation.py`** — DTW and Cosine similarity matrix calculation from GFCC CSVs<br>

**`/model`** — ML model architecture definition<br>
**`GFCC_FACENET.py`** — Training and testing of the ML model<br>
**`GFCC_Facenet.pth`** — Pretrained model weights<br>
**`Model_save.py`** — Saves model embeddings to `.csv` using the pretrained weights<br>
**`Model_Binary_save.py`** — Binarizes embeddings via Median Binarization based on `Model_save.py` output<br>

**`/FuzzyExtractor`**<br>
&nbsp;&nbsp;&nbsp;&nbsp;**`fuzzy_reusable.py`** — Fuzzy extractor implementation using BCH codes<br>
&nbsp;&nbsp;&nbsp;&nbsp;**`fuzzyextractor.py`** — Fuzzy extractor testing + full Hamming distance matrix generation<br>
&nbsp;&nbsp;&nbsp;&nbsp;**`fuzzyextractorsplit.py`** — Hamming distance matrix split by sit/stand posture<br>

**`Threshold.py`** — Accuracy threshold matrix evaluation<br>

---

<ins>_**Default naming is used throughout. If no changes are made, no renaming is needed.**_</ins>

---

## Steps to Simulate

1. Install dependencies: `pip install -r requirements.txt`
   - Only **Python 3.11.11** is supported.
2. Run `gfcc_convert.py` to convert `.wav` audio files to GFCC `.csv` format.
   - Adjust GFCC settings (interval, deltas, etc.) in `gfcc.py` if needed.
3. Run `GFCC_FACENET.py` to train the model and save checkpoints — **or** use the provided `GFCC_Facenet.pth` pretrained weights to skip this step.
4. Run `Model_save.py` to generate embedding CSVs from the model weights.
5. Run `Model_Binary_save.py` to binarize the embeddings (Median Binarization).
6. Run `FuzzyExtractor/fuzzyextractor.py` for the Hamming distance matrix and revocability simulation.
   - Use `fuzzyextractorsplit.py` for sit/stand split matrices.
   - Use `DTW_Calculation.py` for DTW-based and Cosine-based similarity matrices.
7. Run `Threshold.py` to evaluate accuracy thresholds (update matrix name and parameters inside the file).

---

> **Privacy Notice:** Raw `.wav` audio and `.jpg` facial images are not shared due to privacy constraints. End-to-end simulation from raw data is not possible. However, the provided CSV data allows you to run:
> - `DTW_Calculation.py` — DTW and Cosine similarity matrices
> - `FuzzyExtractor/fuzzyextractor.py` and `fuzzyextractorsplit.py` — Hamming distance matrices
> - `Threshold.py` — Threshold and accuracy evaluation
