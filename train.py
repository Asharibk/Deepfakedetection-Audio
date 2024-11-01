import numpy as np
from sklearn.model_selection import train_test_split
from src.data_processing import load_data_from_folders
from src.feature_extraction import extract_spectrograms, extract_features
from src.model import build_model
from src.visualize import plot_training_history

# Load data
real_data, fake_data, labels = load_data_from_folders('data/real', 'data/fake')
sr = 22050

# Extract features
real_spectrograms = extract_spectrograms(real_data, sr)
fake_spectrograms = extract_spectrograms(fake_data, sr)
real_features = [extract_features(s) for s in real_spectrograms]
fake_features = [extract_features(s) for s in fake_spectrograms]
X = np.concatenate((real_features, fake_features), axis=0)
y = np.array([1]*len(real_features) + [0]*len(fake_features))

# Prepare data for training
X_resized = np.array([cv2.resize(x, (128, 128)) for x in X])
X_resized = np.expand_dims(X_resized, axis=-1)
X_train, X_test, y_train, y_test = train_test_split(X_resized, y, test_size=0.2, random_state=38, stratify=y)

# Build and train model
model = build_model(input_shape=(128, 128, 1))
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Plot training history
plot_training_history(history)

# Save model
model.save('audio_deepfake_model.h5')
