import numpy as np
import librosa
from scipy.stats import skew, kurtosis

def extract_spectrogram(audio_data, sr):
    spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sr)
    return spectrogram

def extract_spectrograms(data, sr):
    spectrograms = []
    for audio in data:
        spectrogram = extract_spectrogram(audio, sr)
        spectrograms.append(spectrogram)
    return spectrograms

def extract_features(spectrogram):
    mean = np.mean(spectrogram)
    std_dev = np.std(spectrogram)
    skewness = skew(spectrogram, axis=None)
    kurtosis_val = kurtosis(spectrogram, axis=None)
    return [mean, std_dev, skewness, kurtosis_val]
