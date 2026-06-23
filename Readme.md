# BlowLive

**BlowLive** is a biometric authentication and liveness detection system that uses acoustic blow signals captured via smartphone microphone. It combines GFCC-based feature extraction, a FaceNet-inspired ML model, and fuzzy extractors to authenticate users and detect replay attacks using Doppler shift analysis.

---

## Repository Structure

```
BlowLive/
├── Authentication/          # User authentication pipeline
│   ├── BlowLive-Dataset_Without_Doppler_Shift/   # CSV dataset (50 participants, 10 sessions each)
│   ├── FuzzyExtractor/      # Fuzzy extractor and Hamming distance analysis
│   ├── model/               # ML model architecture
│   ├── GFCC_FACENET.py      # Model training and testing
│   ├── GFCC_Facenet.pth     # Pretrained model weights
│   ├── Model_save.py        # Embedding generation
│   ├── Model_Binary_save.py # Median binarization of embeddings
│   ├── DTW_Calculation.py   # DTW and Cosine similarity matrices
│   ├── Threshold.py         # Accuracy threshold evaluation
│   └── requirements.txt
└── Liveness Detection/      # Replay attack detection pipeline
    ├── BlowLive-Dataset_With_Doppler_Shift/      # WAV dataset (46 users, legitimate + attacker)
    └── liveness_detector.py # Doppler shift-based liveness detector
```

---

## Modules

### Authentication
Authenticates users based on the acoustic signature of their blow signal. Uses GFCC features extracted from audio, passed through a FaceNet-style model, binarized, and secured with a fuzzy extractor.

See [`Authentication/Readme.md`](Authentication/Readme.md) for full setup and simulation steps.

### Liveness Detection
Detects replay attacks by analyzing Doppler shift patterns in the blow signal. Distinguishes between a live user blowing and a pre-recorded playback.

See [`Liveness Detection/README.md`](Liveness%20Detection/README.md) for simulation steps.

---

## Requirements

- Python 3.11.11 (Authentication module)
- Python 3 latest (Liveness Detection module)
- Dependencies: `pip install -r Authentication/requirements.txt`

---

> **Privacy Notice:** Raw `.wav` and `.jpg` biometric data are not publicly shared. The repository includes pre-extracted CSV features sufficient to reproduce the authentication results.
