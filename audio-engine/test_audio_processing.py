
import numpy as np
import pytest

from audio_processing import (
    TARGET_SAMPLE_RATE,
    TARGET_DURATION,
    N_MFCC,
    normalize_audio,
    fix_audio_length,
    extract_mfcc,
    preprocess_audio
)


def test_normalize_audio():
    signal = np.array([0.5, -1.0, 0.25])

    result = normalize_audio(signal)

    assert np.max(np.abs(result)) == 1.0
    assert np.allclose(result, [0.5, -1.0, 0.25])


def test_normalize_silent_audio():
    signal = np.zeros(100)

    result = normalize_audio(signal)

    assert np.array_equal(result, signal)


def test_fix_audio_length():
    short_signal = np.ones(5000)
    long_signal = np.ones(15000)

    short_result = fix_audio_length(short_signal)
    long_result = fix_audio_length(long_signal)

    expected_length = TARGET_SAMPLE_RATE * TARGET_DURATION

    assert len(short_result) == expected_length
    assert len(long_result) == expected_length


def test_extract_mfcc():
    signal = np.random.randn(TARGET_SAMPLE_RATE * TARGET_DURATION)

    mfcc = extract_mfcc(signal)

    assert mfcc.shape[0] == N_MFCC
    assert mfcc.shape[1] > 0


def test_preprocess_audio(tmp_path):
    signal = np.random.randn(TARGET_SAMPLE_RATE * TARGET_DURATION)
    expected_length = TARGET_SAMPLE_RATE * TARGET_DURATION

    import soundfile as sf

    file_path = tmp_path / "test.wav"
    sf.write(file_path, signal, TARGET_SAMPLE_RATE)

    result = preprocess_audio(str(file_path))

    assert result["sample_rate"] == TARGET_SAMPLE_RATE
    assert len(result["signal"]) == expected_length
    assert result["mfcc"].shape[0] == N_MFCC


