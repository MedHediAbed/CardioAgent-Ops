import numpy as np
import librosa


TARGET_SAMPLE_RATE = 2000
TARGET_DURATION = 5  # seconds
N_MFCC = 13


def load_audio(file_path: str):
    """
    Load an audio file and resample it to 2000 Hz.
    """

    signal, sample_rate = librosa.load(
        file_path,
        sr=TARGET_SAMPLE_RATE,
        mono=True
    )

    return signal, sample_rate


def normalize_audio(signal: np.ndarray):
    """
    Normalize the signal between -1 and 1.
    """

    max_value = np.max(np.abs(signal))

    if max_value == 0:
        return signal

    return signal / max_value


def fix_audio_length(signal: np.ndarray):
    """
    Make every audio signal exactly 5 seconds long.
    """

    target_length = TARGET_SAMPLE_RATE * TARGET_DURATION

    # Audio too short -> add zeros
    if len(signal) < target_length:
        padding = target_length - len(signal)

        signal = np.pad(
            signal,
            (0, padding),
            mode="constant"
        )

    # Audio too long -> cut it
    elif len(signal) > target_length:
        signal = signal[:target_length]

    return signal


def extract_mfcc(signal: np.ndarray):
    """
    Extract MFCC features from the audio.
    """

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=TARGET_SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    return mfcc


def preprocess_audio(file_path: str):
    """
    Complete preprocessing pipeline.
    """

    signal, sample_rate = load_audio(file_path)

    signal = normalize_audio(signal)

    signal = fix_audio_length(signal)

    mfcc = extract_mfcc(signal)

    return {
        "signal": signal,
        "sample_rate": sample_rate,
        "mfcc": mfcc
    }