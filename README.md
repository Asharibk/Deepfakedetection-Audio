# Deepfakedetection-Audio

This project detects real vs. fake audio using a CNN-LSTM hybrid model trained on spectrograms.

Model Architecture

This project uses a CNN-LSTM hybrid model:

CNN Layers: Convolutional layers to extract important spatial features from Mel spectrograms.
LSTM Layers: Long Short-Term Memory layers capture sequential dependencies in the audio data.
Dense Layer: Fully connected layer for final classification.
The CNN-LSTM combination leverages both spatial and temporal features, making it effective for distinguishing real from fake audio samples.

Results and Evaluation

After training, the model is evaluated on the test dataset, providing accuracy metrics. Training and validation performance graphs are generated to help diagnose any potential overfitting or underfitting.

Total params:596,993
Trainable params:596,993
Non-trainable params:0




Epochs:100
Accuracy:0.6334
loss:0.6204
Test Accuracy:0.64   






MIT License:

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.





