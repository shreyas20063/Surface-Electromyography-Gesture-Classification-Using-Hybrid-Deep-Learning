"""
SYNAPSE 2026 - Inference Script
================================

This script provides inference capabilities for the trained sEMG gesture
classification model. It supports single file and batch predictions.

Usage:
    # Single file prediction
    python inference.py --input sample.csv

    # Batch prediction
    python inference.py --input data_folder/ --output predictions.csv

    # Use specific fold model
    python inference.py --input sample.csv --fold 3

    # Ensemble prediction (average of all 5 folds)
    python inference.py --input sample.csv --ensemble
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import glob
from scipy.signal import butter, filtfilt, iirnotch, welch
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

SAMPLING_FREQ = 1000
N_CHANNELS = 8
N_CLASSES = 5
N_FEATURES = 144

GESTURE_NAMES = {
    0: "Gesture 0 (G0)",
    1: "Gesture 1 (G1)",
    2: "Gesture 2 (G2)",
    3: "Gesture 3 (G3)",
    4: "Gesture 4 (G4)"
}

# ============================================================================
# SIGNAL PROCESSING
# ============================================================================

def butter_bandpass(lowcut, highcut, fs, order=4):
    """Design a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return b, a

def preprocess_signal(data, fs=1000):
    """
    Apply preprocessing to raw EMG signal.

    Parameters:
    -----------
    data : ndarray
        Raw EMG signal of shape (n_channels, n_samples)
    fs : int
        Sampling frequency in Hz

    Returns:
    --------
    filtered : ndarray
        Preprocessed signal of same shape
    """
    # Bandpass filter: 20-450 Hz
    b, a = butter_bandpass(20, 450, fs, order=4)
    filtered = filtfilt(b, a, data, axis=-1)

    # Notch filters for powerline noise
    for freq in [50, 60]:
        b_notch, a_notch = iirnotch(freq, 30, fs)
        filtered = filtfilt(b_notch, a_notch, filtered, axis=-1)

    return filtered

# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def compute_hjorth_params(x):
    """Compute Hjorth parameters: Activity, Mobility, Complexity."""
    activity = np.var(x)
    dx = np.diff(x)
    dx_var = np.var(dx)
    ddx = np.diff(dx)
    ddx_var = np.var(ddx)
    mobility = np.sqrt(dx_var / (activity + 1e-10))
    mobility_dx = np.sqrt(ddx_var / (dx_var + 1e-10))
    complexity = mobility_dx / (mobility + 1e-10)
    return activity, mobility, complexity

def compute_spectral_features(x, fs=1000):
    """Compute spectral features using Welch's method."""
    freqs, psd = welch(x, fs=fs, nperseg=min(256, len(x)))
    total_power = np.sum(psd)

    # Mean and median frequency
    mean_freq = np.sum(freqs * psd) / (total_power + 1e-10)
    cumsum = np.cumsum(psd)
    median_idx = np.searchsorted(cumsum, total_power / 2)
    median_freq = freqs[min(median_idx, len(freqs)-1)]

    # Spectral entropy
    psd_norm = psd / (total_power + 1e-10)
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))

    # Band powers
    bands = [(20, 50), (50, 100), (100, 200), (200, 450)]
    band_powers = [np.sum(psd[(freqs >= l) & (freqs < h)]) / (total_power + 1e-10)
                   for l, h in bands]

    return [mean_freq, median_freq, spectral_entropy] + band_powers

def compute_ar_coefficients(x, order=2):
    """Compute autoregressive coefficients."""
    n = len(x)
    coeffs = []
    for i in range(1, order + 1):
        if i < n:
            corr = np.correlate(x[i:], x[:-i])[0] / (n - i)
            coeffs.append(corr / (np.var(x) + 1e-10))
        else:
            coeffs.append(0)
    return coeffs

def extract_features(window, fs=1000):
    """
    Extract 144 hand-crafted features from EMG window.

    Parameters:
    -----------
    window : ndarray
        EMG signal of shape (n_channels, n_samples)
    fs : int
        Sampling frequency

    Returns:
    --------
    features : ndarray
        Feature vector of length 144 (18 features x 8 channels)
    """
    features = []

    for ch in range(window.shape[0]):
        x = window[ch]

        # Time-domain features
        mav = np.mean(np.abs(x))
        rms = np.sqrt(np.mean(x**2))
        wl = np.sum(np.abs(np.diff(x)))
        zc = np.sum(np.diff(np.sign(x)) != 0)
        ssc = np.sum(np.diff(np.sign(np.diff(x))) != 0)
        var = np.var(x)
        iemg = np.sum(np.abs(x))
        threshold = 0.01 * np.max(np.abs(x))
        wamp = np.sum(np.abs(np.diff(x)) > threshold)
        log_det = np.exp(np.mean(np.log(np.abs(x) + 1e-10)))

        # Hjorth parameters
        _, mobility, complexity = compute_hjorth_params(x)

        # Spectral features
        spectral = compute_spectral_features(x, fs)

        # AR coefficients
        ar_coeffs = compute_ar_coefficients(x, order=2)

        # Peak-to-peak ratio
        peak_to_peak = (np.max(x) - np.min(x)) / (rms + 1e-10)

        # Combine all features for this channel
        features.extend([mav, rms, wl, zc, ssc, var, iemg, wamp, log_det,
                        mobility, complexity, spectral[0], spectral[1], spectral[2],
                        np.argmax(spectral[3:])/3.0, ar_coeffs[0], ar_coeffs[1], peak_to_peak])

    return np.array(features)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ChannelAttention(nn.Module):
    """Channel attention mechanism for EMG channels."""

    def __init__(self, n_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(n_channels, max(1, n_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, n_channels // reduction), n_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1)

class MultiScaleConv(nn.Module):
    """Multi-scale temporal convolution."""

    def __init__(self, in_ch, out_ch, dropout=0.3):
        super().__init__()
        ch = out_ch // 3
        rem = out_ch - ch * 3

        self.conv_s = nn.Conv1d(in_ch, ch, 3, padding=1)
        self.conv_m = nn.Conv1d(in_ch, ch, 7, padding=3)
        self.conv_l = nn.Conv1d(in_ch, ch + rem, 15, padding=7)

        self.bn = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(F.relu(self.bn(torch.cat([
            self.conv_s(x), self.conv_m(x), self.conv_l(x)
        ], 1))))

class RobustModel(nn.Module):
    """
    Main classification model combining CNN, GRU, and hand-crafted features.

    Architecture:
    - Channel Attention on raw EMG
    - Multi-scale 1D convolutions
    - Bidirectional GRU with temporal attention
    - Feature branch for hand-crafted features
    - Fusion and classification
    """

    def __init__(self, n_channels=8, n_classes=5, n_features=144, dropout=0.35):
        super().__init__()

        # Signal processing branch
        self.channel_attn = ChannelAttention(n_channels, reduction=2)
        self.ms_conv = MultiScaleConv(n_channels, 36, dropout=dropout*0.7)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Sequential(
            nn.Conv1d(36, 48, 5, padding=2),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Dropout(dropout)
        )

        self.gru = nn.GRU(48, 36, batch_first=True, bidirectional=True)

        self.temp_attn = nn.Sequential(
            nn.Linear(72, 18),
            nn.Tanh(),
            nn.Linear(18, 1, bias=False)
        )

        # Feature branch
        self.feat_branch = nn.Sequential(
            nn.Linear(n_features, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(48, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(72 + 32, 52),
            nn.BatchNorm1d(52),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(52, n_classes)
        )

    def forward(self, x_raw, x_feat):
        # Signal branch
        x = self.channel_attn(x_raw)
        x = self.pool1(self.ms_conv(x))
        x = self.conv2(x)

        x = x.permute(0, 2, 1)
        gru_out, _ = self.gru(x)

        attn = F.softmax(self.temp_attn(gru_out), dim=1)
        context = torch.sum(gru_out * attn, dim=1)

        # Feature branch
        feat_out = self.feat_branch(x_feat)

        # Fusion and classification
        return self.classifier(torch.cat([context, feat_out], dim=1))

# ============================================================================
# INFERENCE ENGINE
# ============================================================================

class GestureClassifier:
    """
    High-level inference class for gesture classification.

    Example:
    --------
    classifier = GestureClassifier(model_dir='./models')
    prediction = classifier.predict('sample.csv')
    print(f"Predicted gesture: {prediction['gesture']}")
    """

    def __init__(self, model_dir='.', fold=None, ensemble=True, device=None):
        """
        Initialize the classifier.

        Parameters:
        -----------
        model_dir : str
            Directory containing model checkpoints and scalers
        fold : int, optional
            Specific fold to use (1-5). If None and ensemble=False, uses fold 1
        ensemble : bool
            Whether to use ensemble of all 5 folds
        device : str, optional
            Device to use ('cuda' or 'cpu'). Auto-detected if None
        """
        self.model_dir = model_dir
        self.ensemble = ensemble
        self.fold = fold if fold else 1

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        self.models = []
        self.scalers_raw = []
        self.scalers_feat = []

        self._load_models()

    def _load_models(self):
        """Load model(s) and scalers."""
        if self.ensemble:
            folds_to_load = range(1, 6)
            print("Loading ensemble (5 folds)...")
        else:
            folds_to_load = [self.fold]
            print(f"Loading single model (fold {self.fold})...")

        for fold_idx in folds_to_load:
            # Load model
            model_path = os.path.join(self.model_dir, f'model_fold{fold_idx}.pth')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found: {model_path}")

            checkpoint = torch.load(model_path, map_location=self.device)
            model = RobustModel(N_CHANNELS, N_CLASSES, N_FEATURES, dropout=0.35)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            self.models.append(model)

            # Load scalers
            scaler_raw_path = os.path.join(self.model_dir, f'scaler_raw_fold{fold_idx}.pkl')
            scaler_feat_path = os.path.join(self.model_dir, f'scaler_feat_fold{fold_idx}.pkl')

            with open(scaler_raw_path, 'rb') as f:
                self.scalers_raw.append(pickle.load(f))

            with open(scaler_feat_path, 'rb') as f:
                self.scalers_feat.append(pickle.load(f))

        print(f"Loaded {len(self.models)} model(s)")

    def _load_emg_file(self, file_path):
        """Load EMG data from CSV file."""
        df = pd.read_csv(file_path)
        raw_data = df.values.T

        # Ensure shape is (8, n_samples)
        if raw_data.shape[0] != 8:
            if raw_data.shape[1] == 8:
                raw_data = raw_data.T
            else:
                raise ValueError(f"Expected 8 channels, got shape {raw_data.shape}")

        return raw_data

    def predict(self, input_path):
        """
        Predict gesture from EMG file.

        Parameters:
        -----------
        input_path : str
            Path to CSV file containing EMG data

        Returns:
        --------
        dict with keys:
            - 'gesture': int (0-4)
            - 'gesture_name': str
            - 'confidence': float (0-1)
            - 'probabilities': list of 5 floats
        """
        # Load and preprocess
        raw_data = self._load_emg_file(input_path)
        processed = preprocess_signal(raw_data, SAMPLING_FREQ)
        features = extract_features(processed, SAMPLING_FREQ)

        all_probs = []

        for i, model in enumerate(self.models):
            # Scale
            X_raw_scaled = self.scalers_raw[i].transform(processed.flatten().reshape(1, -1))
            X_feat_scaled = self.scalers_feat[i].transform(features.reshape(1, -1))

            # Reshape for model
            X_raw_tensor = torch.FloatTensor(X_raw_scaled).reshape(1, N_CHANNELS, -1).to(self.device)
            X_feat_tensor = torch.FloatTensor(X_feat_scaled).to(self.device)

            # Predict
            with torch.no_grad():
                outputs = model(X_raw_tensor, X_feat_tensor)
                probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
                all_probs.append(probs)

        # Average probabilities for ensemble
        avg_probs = np.mean(all_probs, axis=0)
        predicted_class = int(np.argmax(avg_probs))
        confidence = float(avg_probs[predicted_class])

        return {
            'gesture': predicted_class,
            'gesture_name': GESTURE_NAMES[predicted_class],
            'confidence': confidence,
            'probabilities': avg_probs.tolist()
        }

    def predict_batch(self, input_dir, output_file=None):
        """
        Predict gestures for all CSV files in a directory.

        Parameters:
        -----------
        input_dir : str
            Directory containing CSV files
        output_file : str, optional
            Path to save predictions CSV

        Returns:
        --------
        DataFrame with predictions
        """
        csv_files = glob.glob(os.path.join(input_dir, '*.csv'))

        if not csv_files:
            raise ValueError(f"No CSV files found in {input_dir}")

        results = []

        for file_path in csv_files:
            try:
                pred = self.predict(file_path)
                results.append({
                    'file': os.path.basename(file_path),
                    'gesture': pred['gesture'],
                    'gesture_name': pred['gesture_name'],
                    'confidence': pred['confidence'],
                    **{f'prob_G{i}': pred['probabilities'][i] for i in range(N_CLASSES)}
                })
                print(f"  {os.path.basename(file_path)}: {pred['gesture_name']} ({pred['confidence']:.2%})")
            except Exception as e:
                print(f"  Error processing {file_path}: {e}")
                results.append({
                    'file': os.path.basename(file_path),
                    'gesture': -1,
                    'gesture_name': 'ERROR',
                    'confidence': 0.0
                })

        df = pd.DataFrame(results)

        if output_file:
            df.to_csv(output_file, index=False)
            print(f"\nPredictions saved to: {output_file}")

        return df

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='SYNAPSE 2026 - sEMG Gesture Classification Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file prediction
  python inference.py --input sample.csv

  # Batch prediction
  python inference.py --input data_folder/ --output predictions.csv

  # Use specific fold
  python inference.py --input sample.csv --fold 3

  # Ensemble prediction (default)
  python inference.py --input sample.csv --ensemble
        """
    )

    parser.add_argument('--input', '-i', required=True,
                        help='Input CSV file or directory')
    parser.add_argument('--output', '-o', default=None,
                        help='Output CSV file for batch predictions')
    parser.add_argument('--model-dir', '-m', default='.',
                        help='Directory containing model files (default: current)')
    parser.add_argument('--fold', '-f', type=int, choices=[1, 2, 3, 4, 5],
                        help='Use specific fold model (1-5)')
    parser.add_argument('--ensemble', '-e', action='store_true', default=True,
                        help='Use ensemble of all 5 folds (default)')
    parser.add_argument('--no-ensemble', action='store_true',
                        help='Disable ensemble, use single fold')
    parser.add_argument('--device', '-d', choices=['cuda', 'cpu'],
                        help='Device to use (auto-detected if not specified)')

    args = parser.parse_args()

    # Handle ensemble flag
    use_ensemble = args.ensemble and not args.no_ensemble
    if args.fold:
        use_ensemble = False

    print("=" * 60)
    print("SYNAPSE 2026 - sEMG Gesture Classification")
    print("=" * 60)
    print()

    # Initialize classifier
    classifier = GestureClassifier(
        model_dir=args.model_dir,
        fold=args.fold,
        ensemble=use_ensemble,
        device=args.device
    )

    print()

    # Check if input is file or directory
    if os.path.isfile(args.input):
        # Single file prediction
        print(f"Processing: {args.input}")
        result = classifier.predict(args.input)

        print()
        print("=" * 40)
        print("PREDICTION RESULT")
        print("=" * 40)
        print(f"  Gesture: {result['gesture']} ({result['gesture_name']})")
        print(f"  Confidence: {result['confidence']:.2%}")
        print()
        print("  Class Probabilities:")
        for i, prob in enumerate(result['probabilities']):
            bar = '#' * int(prob * 30)
            print(f"    G{i}: {prob:6.2%} |{bar}")

    elif os.path.isdir(args.input):
        # Batch prediction
        print(f"Processing directory: {args.input}")
        print()
        df = classifier.predict_batch(args.input, args.output)

        print()
        print("=" * 40)
        print("BATCH SUMMARY")
        print("=" * 40)
        print(f"  Total files: {len(df)}")
        print(f"  Successful: {len(df[df['gesture'] >= 0])}")

        if len(df[df['gesture'] >= 0]) > 0:
            print("\n  Gesture distribution:")
            for i in range(N_CLASSES):
                count = len(df[df['gesture'] == i])
                print(f"    G{i}: {count} ({count/len(df)*100:.1f}%)")
    else:
        print(f"Error: Input not found: {args.input}")
        sys.exit(1)

if __name__ == '__main__':
    main()
