import os
from typing import List, Tuple
import numpy as np
import librosa
from scipy import fft


# IMFCC implementation
def compute_imfcc(y: np.ndarray, sr: int = 22050, n_imfcc: int = 13, **kwargs):
    """
    Extract Inverted Mel-Frequency Cepstral Coefficients
    """
    n_fft = kwargs.get("n_fft", 2048)

    S = np.abs(librosa.stft(y, n_fft=n_fft, **kwargs))
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=128)
    mel_basis_inverted = mel_basis[::-1, :]
    mel_spectrogram = np.dot(mel_basis_inverted, S)

    log_mel_spectrogram = np.log(mel_spectrogram + 1e-9)

    imfccs = fft.dct(log_mel_spectrogram, axis=0, norm="ortho")[:n_imfcc]

    return imfccs


# CQCC implementation
def compute_cqcc(
    y: np.ndarray,
    sr: int = 22050,
    n_cqcc: int = 13,
    hop_length: int = 512,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    fmin: any = None,
):
    """
    Extract Constant-Q Cepstral Coefficients
    """
    C = np.abs(
        librosa.cqt(
            y,
            sr=sr,
            hop_length=hop_length,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            fmin=fmin,
        )
    )

    log_C = np.log(C + 1e-9)

    cqccs = fft.dct(log_C, axis=0, norm="ortho")[:n_cqcc]

    return cqccs


def extract_features_from_array(y: np.ndarray, sr: float) -> np.ndarray:
    """
    Extracts audio features from a loaded audio sample.

    Args:
        y: Audio time series array
        sr: Sample rate of the audio

    Returns:
        Array of extracted audio features
    """

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    imfccs = compute_imfcc(y=y, sr=sr, n_imfcc=13)

    features = np.hstack(
        [
            np.mean(chroma_stft),
            np.mean(rms),
            np.mean(spectral_centroid),
            np.mean(spectral_bandwidth),
            np.mean(rolloff),
            np.mean(zero_crossing_rate),
            np.mean(tonnetz),
            np.mean(spectral_contrast),
            np.mean(mfccs, axis=1),
            np.mean(imfccs, axis=1),
        ]
    )
    return features


def load_and_process_audio(
    file_path: str, target_sr: int = 22050
) -> Tuple[np.ndarray, float]:
    """
    Loads audio file and resamples to target frequency.

    Args:
        file_path: Path to the audio file
        target_sr: Target sample rate in Hz (default: 22050)

    Returns:
        Audio data array and its sample rate
    """
    try:
        y, sr = librosa.load(file_path, sr=target_sr)
        return y, sr
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return np.array([]), target_sr


def process_audio_folder(
    folder_path: str, label: int, target_sr: int = 22050
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Processes all audio files in a REAL or FAKE folder.

    Args:
        folder_path: Directory containing audio files
        label: Class label (0 or 1) for the audio files
        target_sr: Target sample rate in Hz (default: 22050)

    Returns:
        List of processed audio features and their labels
    """
    features_list = []
    labels_list = []
    audio_files = []

    for f in os.listdir(folder_path):
        audio_files.append(f)

    print(f"Processing {len(audio_files)} files in {folder_path}...")

    for audio_file in audio_files:
        file_path = os.path.join(folder_path, audio_file)

        y, sr = load_and_process_audio(file_path, target_sr)

        if len(y) > 0:
            features = extract_features_from_array(y, sr)

            features_list.append(features)
            labels_list.append(label)

    return features_list, labels_list


def extract_all_dataset(path: str, save: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts datasets from specified path for training.

    Args:
        path: Location of raw data files
        save: Whether to save processed data (default: True)

    Returns:
        Features and labels as numpy arrays
    """
    all_features = []
    all_labels = []

    # Get all subdirectories in the path (subdataset_1, subdataset_2, etc.)
    dataset_dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]

    for dataset_dir in dataset_dirs:
        dataset_path = os.path.join(path, dataset_dir)
        print(f"Processing dataset directory: {dataset_path}")

        # Get REAL and FAKE folders within each subdataset
        class_dirs = [
            d
            for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ]

        for class_dir in class_dirs:
            class_path = os.path.join(dataset_path, class_dir)
            label = 1 if class_dir.upper() == "REAL" else 0

            # Pass the target_sr parameter (define a default value if not already defined elsewhere)
            target_sr = 22050  # Common sampling rate, adjust as needed

            # Call the process_audio_folder function with the correct parameters
            features, labels = process_audio_folder(
                folder_path=class_path, target_sr=target_sr, label=label
            )

            all_features.extend(features)
            all_labels.extend(labels)

    all_features = np.array(all_features)
    all_labels = np.array(all_labels)

    if save:
        # Add code to save the processed data
        np.save("features.npy", all_features)
        np.save("labels.npy", all_labels)
        print(
            f"Saved processed data: features shape {all_features.shape}, labels shape {all_labels.shape}"
        )

    return all_features, all_labels
