import customtkinter as ctk
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps, ImageTk
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import io

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class DigitRecognizerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Magnificent Handwritten Digit Recognizer")
        self.geometry("800x500")
        self.resizable(False, False)
        self.model = self.load_model()
        self.create_widgets()
        self.image1 = Image.new("L", (280, 280), color=0)
        self.draw = ImageDraw.Draw(self.image1)
        self.tk_preview = None
        self.tk_chart = None

    def load_model(self):
        try:
            model = keras.models.load_model("mnist_cnn.h5")
        except:
            (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
            x_train = x_train.astype("float32") / 255.0
            x_test = x_test.astype("float32") / 255.0
            x_train = np.expand_dims(x_train, -1)
            x_test = np.expand_dims(x_test, -1)
            model = keras.Sequential([
                keras.Input(shape=(28,28,1)),
                keras.layers.Conv2D(32, 3, activation='relu'),
                keras.layers.Conv2D(64, 3, activation='relu'),
                keras.layers.MaxPooling2D(),
                keras.layers.Dropout(0.25),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(10, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(x_train, y_train, epochs=11, validation_data=(x_test, y_test))
            model.save("mnist_cnn.h5")
        return model

    def create_widgets(self):
        self.left_frame = ctk.CTkFrame(self, width=300, height=400)
        self.left_frame.place(x=30, y=30)
        self.right_frame = ctk.CTkFrame(self, width=420, height=400)
        self.right_frame.place(x=350, y=30)

        self.canvas = tk.Canvas(self.left_frame, width=280, height=280, bg="black", cursor="cross")
        self.canvas.pack(pady=20)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<Button-1>", self.paint)

        self.clear_btn = ctk.CTkButton(self.left_frame, text="Clear", command=self.clear_canvas)
        self.clear_btn.pack(pady=10)
        self.predict_btn = ctk.CTkButton(self.left_frame, text="Predict", command=self.predict_digit)
        self.predict_btn.pack(pady=10)

        self.pred_label = ctk.CTkLabel(self.right_frame, text="Draw a digit and click Predict!", font=("Arial", 24))
        self.pred_label.pack(pady=20)
        self.confidence_label = ctk.CTkLabel(self.right_frame, text="", font=("Arial", 18))
        self.confidence_label.pack(pady=10)
        self.img_label = ctk.CTkLabel(self.right_frame, text="Input Preview:")
        self.img_label.pack(pady=5)
        self.preview_canvas = tk.Canvas(self.right_frame, width=56, height=56, bg="white")
        self.preview_canvas.pack(pady=5)
        self.plot_label = ctk.CTkLabel(self.right_frame, text="Confidence Bar Chart:")
        self.plot_label.pack(pady=5)
        self.plot_canvas = tk.Canvas(self.right_frame, width=400, height=100, bg="white")
        self.plot_canvas.pack(pady=5)

    def paint(self, event):
        x1, y1 = (event.x - 8), (event.y - 8)
        x2, y2 = (event.x + 8), (event.y + 8)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw.ellipse([x1, y1, x2, y2], fill=255)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image1 = Image.new("L", (280, 280), color=0)
        self.draw = ImageDraw.Draw(self.image1)
        self.preview_canvas.delete("all")
        self.plot_canvas.delete("all")
        self.pred_label.configure(text="Draw a digit and click Predict!")
        self.confidence_label.configure(text="")
        self.tk_preview = None
        self.tk_chart = None

    def preprocess_image(self, img):
        # Invert, crop to bounding box, pad, resize, normalize
        img = ImageOps.invert(img)
        arr = np.array(img)
        # Threshold to binary
        arr = (arr > 20).astype(np.uint8) * 255
        img = Image.fromarray(arr)
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
        # Pad to square
        max_side = max(img.size)
        new_img = Image.new('L', (max_side, max_side), 0)
        new_img.paste(img, ((max_side - img.size[0]) // 2, (max_side - img.size[1]) // 2))
        img = new_img.resize((28, 28), Image.LANCZOS)
        arr = np.array(img).astype('float32') / 255.0
        return arr, img

    def predict_digit(self):
        arr, img = self.preprocess_image(self.image1)
        img_arr = np.expand_dims(arr, axis=(0, -1))
        pred = self.model.predict(img_arr)[0]
        pred_digit = np.argmax(pred)
        confidence = pred[pred_digit]
        self.pred_label.configure(text=f"Prediction: {pred_digit}", text_color="#4F8BF9")
        self.confidence_label.configure(text=f"Confidence: {confidence*100:.2f}%")
        # Show preview
        preview_img = img.resize((56, 56))
        self.preview_canvas.delete("all")
        self.tk_preview = ImageTk.PhotoImage(preview_img)
        self.preview_canvas.create_image(0, 0, anchor="nw", image=self.tk_preview)
        # Show confidence bar chart
        self.plot_canvas.delete("all")
        fig, ax = plt.subplots(figsize=(4, 1))
        ax.bar(range(10), pred, color="#4F8BF9")
        ax.set_xticks(range(10))
        ax.set_yticks([])
        ax.set_xlabel("Digit")
        ax.set_title("Prediction Probabilities", fontsize=10)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        chart_img = Image.open(buf)
        chart_img = chart_img.resize((400, 100))
        self.tk_chart = ImageTk.PhotoImage(chart_img)
        self.plot_canvas.create_image(0, 0, anchor="nw", image=self.tk_chart)
        # Show MNIST sample for comparison
        self.show_mnist_sample(pred_digit)

    def show_mnist_sample(self, digit):
        (x_train, y_train), _ = keras.datasets.mnist.load_data()
        idx = np.where(y_train == digit)[0][0]
        mnist_img = x_train[idx]
        mnist_img = Image.fromarray(mnist_img).resize((56, 56), Image.LANCZOS)
        win = tk.Toplevel(self)
        win.title(f"MNIST Sample for {digit}")
        win.geometry("120x70")
        tk.Label(win, text="Your Input").grid(row=0, column=0)
        tk.Label(win, text="MNIST").grid(row=0, column=1)
        user_img = self.tk_preview
        mnist_img_tk = ImageTk.PhotoImage(mnist_img)
        tk.Label(win, image=user_img).grid(row=1, column=0)
        tk.Label(win, image=mnist_img_tk).grid(row=1, column=1)
        win.after(3000, win.destroy)  # Auto-close after 3 seconds
        win.mainloop()

if __name__ == "__main__":
    app = DigitRecognizerApp()
    app.mainloop()
