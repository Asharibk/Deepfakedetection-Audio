import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

def load_audio_file(file_path, duration=2.0, sr=22050):
    audio_data, sr = librosa.load(file_path, sr=sr, duration=duration, mono=True)
    return audio_data, sr

def load_data_from_folders(real_folder, fake_folder, sr=22050):
    real_data = []
    fake_data = []
    labels = []

    # Load real data
    for file_name in os.listdir(real_folder):
        if file_name.endswith('.wav'):
            audio_data, _ = load_audio_file(os.path.join(real_folder, file_name), sr=sr)
            real_data.append(audio_data)
            labels.append(1)  # Label 1 for real voices

    # Load fake data
    for file_name in os.listdir(fake_folder):
        if file_name.endswith('.wav'):
            audio_data, _ = load_audio_file(os.path.join(fake_folder, file_name), sr=sr)
            fake_data.append(audio_data)
            labels.append(0)  # Label 0 for fake voices

    return real_data, fake_data, labels
