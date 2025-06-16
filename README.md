# 🔢 Magnificent Handwritten Digit Recognizer

A modern, visually stunning Python GUI application that recognizes handwritten digits (0–9) using a deep neural network trained on the MNIST dataset.

Draw a digit, hit **Predict**, and instantly get:
- Model prediction
- Confidence bar chart
- Side-by-side comparison with real MNIST samples

All inside a beautiful, dark-themed interface powered by `customtkinter`.

---

## ✨ Features

✍️ **Draw digits** directly on a sleek canvas  
🤖 **Deep Learning** model (CNN) trained on MNIST  
📊 **Real-time confidence chart** for predictions  
🖼️ **Preview & compare** with real MNIST digit samples  
🌙 **Dark-themed GUI** using `customtkinter`  
💾 **Auto model training & saving** on first run

---

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/magnificent-digit-recognizer.git
cd magnificent-digit-recognizer
```
2. **Install dependencies**
```bash
pip install -r requirements.txt
```
3. **Run the application**
```bash
python digit_gui.py
```
---

🎨 **How to Use**

Draw a digit (0–9) on the canvas.

Click Predict to:

View the predicted digit

See a confidence score bar chart

Compare your digit with a real MNIST sample

Click Clear to reset the canvas.

---

🧠 **How It Works**
The app uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

It is trained on the MNIST dataset for 11 epochs (automatically on the first run).

Your input is preprocessed to match the MNIST digit style:

Resized to 28x28 pixels

Grayscale converted

Centered and padded

Normalized to values between 0 and 1

The model then:

Predicts the digit

Displays a confidence chart

Shows an actual MNIST sample of the predicted digit for comparison

---

🛠️ **Customization Tips**
Want to improve recognition accuracy for your handwriting?

Collect your own digit samples using the drawing canvas

Modify the model architecture or training parameters (e.g., number of epochs) in digit_gui.py

Retrain the model with your data for better personalization

---

📋 **Requirements**
Python 3.8+

Required packages:

tensorflow

customtkinter

matplotlib

numpy

Pillow

(Full list available in requirements.txt)

---

🤝 **Contributing**
Pull requests and suggestions are welcome!
If you have an idea or find a bug, feel free to open an issue.
