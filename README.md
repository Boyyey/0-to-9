# 0-to-9

Magnificent Handwritten Digit Recognizer
A beautiful, modern Python GUI app that recognizes handwritten digits (0–9) using a deep neural network trained on the MNIST dataset. Draw a digit, get instant predictions, confidence scores, and compare your input to real MNIST samples—all in a visually stunning interface!

Features
✍️ Draw digits directly on a modern canvas
🤖 Deep learning model (CNN) trained on MNIST
📊 Confidence bar chart for predictions
🖼️ Input preview and side-by-side comparison with real MNIST samples
🌙 Beautiful, dark-themed GUI using customtkinter
🔄 Automatic model training and saving

Installation

Clone the repository:

git clone https://github.com/yourusername/magnificent-digit-recognizer.git

cd magnificent-digit-recognizer

Install dependencies:

pip install -r requirements.txt

Run the app:

python digit_gui.py

Usage
Draw a digit (0–9) in the left canvas.
Click Predict to see the model’s prediction and confidence.
View your input, the confidence bar chart, and a real MNIST sample for comparison.
Click Clear to reset the canvas.

How It Works
The app uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras.
The model is trained on the MNIST dataset for 11 epochs (automatically on first run).
Your drawing is preprocessed (centered, padded, resized, normalized) to match MNIST style before prediction.
The app displays a side-by-side comparison of your input and a real MNIST digit for the predicted class.

Customization
To improve accuracy for your handwriting, consider collecting your own digit samples and retraining the model.
You can adjust the number of training epochs or model architecture in digit_gui.py.

Requirements
Python 3.8+
See requirements.txt for all dependencies.
