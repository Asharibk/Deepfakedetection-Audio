import os
import cv2
import numpy as np
import librosa
import tensorflow as tf
import librosa.display
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, LSTM, MaxPooling2D, Dropout, TimeDistributed, Reshape
from tensorflow.keras.utils import to_categorical
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
real_data, fake_data, labels = load_data_from_folders('/kaggle/input/in-the-wild-audio-deepfake/release_in_the_wild/real', '/kaggle/input/in-the-wild-audio-deepfake/release_in_the_wild/fake')
def visualize_waveform(audio_data, sr, title):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio_data, sr=sr)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()
def visualize_audio_from_folders(real_folder, fake_folder):
    # Load and visualize real data
    real_files = os.listdir(real_folder)[:2]
    for file_name in real_files:
        if file_name.endswith('.wav'):
            audio_data, sr = load_audio_file(os.path.join(real_folder, file_name))
            visualize_waveform(audio_data, sr, title=f'Real: {file_name}')
    
    # Load and visualize fake data
    fake_files = os.listdir(fake_folder)[:2]
    for file_name in fake_files:
        if file_name.endswith('.wav'):
            audio_data, sr = load_audio_file(os.path.join(fake_folder, file_name))
            visualize_waveform(audio_data, sr, title=f'Fake: {file_name}')

visualize_audio_from_folders('/kaggle/input/in-the-wild-audio-deepfake/release_in_the_wild/real', '/kaggle/input/in-the-wild-audio-deepfake/release_in_the_wild/fake')
def extract_spectrogram(audio_data, sr):
    spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sr)
    return spectrogram
def extract_spectrograms(data, sr):
    spectrograms = []
    for audio in data:
        spectrogram = extract_spectrogram(audio, sr)
        spectrograms.append(spectrogram)
    return spectrograms
sr = 22050
real_spectrograms = extract_spectrograms(real_data, sr)
fake_spectrograms = extract_spectrograms(fake_data, sr)
def visualize_spectrogram(spectrogram, sr):
    # Plot the spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.show()
# Visualize first two real spectrograms
for spectrogram in real_spectrograms[:2]:
    visualize_spectrogram(spectrogram, sr)
# Visualize first two fake spectrograms
for spectrogram in fake_spectrograms[:2]:
    visualize_spectrogram(spectrogram, sr)
def extract_features(spectrogram):
    mean = np.mean(spectrogram)
    std_dev = np.std(spectrogram)
    skewness = skew(spectrogram, axis=None)
    kurtosis_val = kurtosis(spectrogram, axis=None)
    return [mean, std_dev, skewness, kurtosis_val]
# Extract features from spectrograms
real_features = [extract_features(spectrogram) for spectrogram in real_spectrograms]
fake_features = [extract_features(spectrogram) for spectrogram in fake_spectrograms]
labels
real_features
X = np.concatenate((real_features, fake_features), axis=0)
y = np.array([1]*len(real_features) + [0]*len(fake_features))
X_resized = np.array([cv2.resize(x, (128, 128)) for x in X])
X_resized = np.expand_dims(X_resized, axis=-1)
X_train, X_test, y_train, y_test = train_test_split(X_resized, y, test_size=0.2, random_state=38, stratify=y)
# Build the hybrid CNN-RNN model
model = Sequential()

# Add CNN layers
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

# Prepare for RNN
model.add(Reshape((32, 2048)))  # Adjust this shape according to your architecture

# Add RNN layers
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64))

# Add dense layers for classification
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))  # Use sigmoid for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
