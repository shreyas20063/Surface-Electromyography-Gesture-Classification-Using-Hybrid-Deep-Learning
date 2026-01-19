
# sEMG Gesture Classification

---

## Overview

This repository presents a surface electromyography (sEMG)–based hand gesture classification system using hybrid deep learning. The proposed model classifies **5 distinct hand gestures** from **8-channel EMG signals** sampled at **1000 Hz**, combining convolutional, recurrent, and hand-crafted feature–based representations.

**Authors**

* Duggimpudi Shreyas Reddy
* Rajat Gupta
* Sudhakar S. Dalwayi

### Key Results

* **Overall Accuracy**: 74.29% ± 3.15%
* **Model Parameters**: 45,781 (lightweight)
* **Validation**: 5-Fold Subject-Grouped Cross-Validation
* **Per-Class Accuracy**:

  * G0: 64.57% ± 11.32%
  * G1: 72.57% ± 8.92%
  * G2: 68.95% ± 14.53%
  * G3: 71.43% ± 5.96%
  * G4: 93.90% ± 4.37%

---

## Repository Structure

```
v11_final/
|-- README.md                 # Project documentation
|-- train.py                  # Training script with 5-fold CV
|-- inference.py              # Inference script for predictions
|-- requirements.txt          # Python dependencies
|-- cv_results_v11.png        # Cross-validation visualization
|-- cv_summary_v11.json       # Training results summary
|
|-- Model Artifacts (5 folds):
|   |-- model_fold1.pth
|   |-- model_fold2.pth
|   |-- model_fold3.pth
|   |-- model_fold4.pth
|   |-- model_fold5.pth
|
|-- Scalers (for preprocessing):
|   |-- scaler_raw_fold1.pkl
|   |-- scaler_feat_fold1.pkl
|   |-- (... for all 5 folds)
|
|-- report/
    |-- report.tex            # LaTeX technical report
    |-- report.pdf            # Compiled report
```

---

## Quick Start

### 1. Installation

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### 2. Training (Optional – pretrained models included)

```bash
python train.py
```

This performs 5-fold cross-validation and saves:

* Model checkpoints
* Preprocessing scalers
* Performance visualizations
* Summary metrics

### 3. Inference

```bash
# Single file prediction
python inference.py --input path/to/emg_sample.csv

# Batch prediction
python inference.py --input path/to/data_folder/ --output predictions.csv

# Specify a fold model
python inference.py --input sample.csv --fold 1
```

---

## Model Architecture

### RobustModel (45,781 parameters)

```
Input: 8-channel EMG signal (8 × 1000 samples) + 144 hand-crafted features

Signal Branch:
  - Channel Attention
  - Multi-Scale Conv1D (kernels: 3, 7, 15)
  - MaxPool1D
  - Conv1D
  - MaxPool1D
  - Bidirectional GRU
  - Temporal Attention

Feature Branch:
  - Fully Connected Layers with BatchNorm and ReLU

Fusion:
  - Feature Concatenation
  - Fully Connected Layers
  - Softmax Classification (5 classes)
```

### Key Components

1. Channel-wise attention for adaptive sensor weighting
2. Multi-scale temporal convolutions
3. Bidirectional GRU for sequence modeling
4. Temporal attention for discriminative segment focus
5. Fusion of learned and hand-crafted features

---

## Feature Engineering

### 144 Hand-Crafted Features (18 per channel × 8 channels)

* **Time-Domain**: MAV, RMS, WL, ZC, SSC, VAR, IEMG, WAMP, Log Detector
* **Hjorth Parameters**: Mobility, Complexity
* **Spectral Features**: Mean frequency, median frequency, entropy, band powers
* **AR Coefficients**: AR(1), AR(2)
* **Other**: Peak-to-peak ratio

### Preprocessing

1. Bandpass filtering (20–450 Hz)
2. Powerline noise removal (50/60 Hz notch)
3. Robust normalization

---

## Training Details

* **Data Augmentation**: noise injection, scaling, temporal shifting, mixup
* **Class Balancing**: weighted sampling, focal loss
* **Optimizer**: AdamW
* **Scheduler**: Cosine annealing with warmup
* **Regularization**: gradient clipping, early stopping

---

## Dataset Information

* **Subjects**: 25
* **Sessions**: 3 per subject
* **Gestures**: 5 classes
* **Trials**: 7 per gesture per session
* **Total Samples**: 2,625
* **Sampling Rate**: 1000 Hz
* **Channels**: 8 EMG electrodes

---

## Results Visualization

The provided visualization includes:

* Confusion matrices
* Per-class performance across folds
* Mean ± standard deviation accuracy

---

## Hardware Requirements

* **Training**: GPU recommended
* **Inference**: CPU sufficient
* **RAM**: ≥ 8 GB
* **Storage**: ~100 MB

---

## Dependencies

Key libraries include:

* PyTorch
* NumPy
* SciPy
* scikit-learn
* pandas
* matplotlib
* seaborn

See `requirements.txt` for the complete list.

---

## Citation

If you use this work, please cite:

```
@misc{semg_hybrid_2026,
  title={Hybrid Deep Learning for sEMG-Based Hand Gesture Classification},
  author={Duggimpudi Shreyas Reddy and Rajat Gupta and Sudhakar S. Dalwayi},
  year={2026}
}
```

---

## License

This project is released for academic and research use.

