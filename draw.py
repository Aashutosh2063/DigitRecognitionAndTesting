import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw

model = np.load("trained_network.npz")

weights1 = model["weights1"]
biases1 = model["biases1"]
weights2 = model["weights2"]
biases2 = model["biases2"]

print("Neural network loaded!")

root = tk.Tk()
root.title("My Neural Network")
root.geometry("500x650")
root.configure(bg="#1e1e2e")

title = tk.Label(
    root,
    text="Draw a Digit",
    font=("Arial", 24, "bold"),
    bg="#1e1e2e",
    fg="white"
)
title.pack(pady=20)

canvas = tk.Canvas(
    root,
    width=280,
    height=280,
    bg="black",
    highlightthickness=2,
    highlightbackground="#89b4fa"
)
canvas.pack(pady=10)

image = Image.new("L", (280, 280), 0)
draw = ImageDraw.Draw(image)

def paint(event):
    x, y = event.x, event.y
    radius = 10

    canvas.create_oval(
        x - radius, y - radius,
        x + radius, y + radius,
        fill="white",
        outline="white"
    )

    draw.ellipse(
        (x - radius, y - radius,
         x + radius, y + radius),
        fill=255
    )


canvas.bind("<B1-Motion>", paint)

def predict_digit():

    bbox = image.getbbox()

    if bbox is None:
        result_label.config(text="Draw a digit first!")
        probabilities_label.config(text="")
        return

    cropped = image.crop(bbox)

    cropped.thumbnail((20, 20))

    resized = Image.new("L", (28, 28), 0)

    x_offset = (28 - cropped.width) // 2
    y_offset = (28 - cropped.height) // 2

    resized.paste(cropped, (x_offset, y_offset))

    x = np.array(resized, dtype=np.float32).flatten() / 255.0

    z1 = weights1 @ x + biases1
    a1 = np.maximum(0, z1)

    z2 = weights2 @ a1 + biases2

    exp_scores = np.exp(z2 - np.max(z2))
    probabilities = exp_scores / np.sum(exp_scores)

    predicted_digit = np.argmax(probabilities)
    confidence = probabilities[predicted_digit] * 100

    result_label.config(
        text=f"Prediction: {predicted_digit} ({confidence:.1f}%)"
    )

    probability_text = "\n".join(
        f"{digit}: {probabilities[digit] * 100:.1f}%"
        for digit in range(10)
    )

    probabilities_label.config(text=probability_text)

def clear_canvas():
    canvas.delete("all")
    draw.rectangle((0, 0, 280, 280), fill=0)
    result_label.config(text="Prediction: ?")
    probabilities_label.config(text="")


button_frame = tk.Frame(root, bg="#1e1e2e")
button_frame.pack(pady=10)

clear_button = tk.Button(
    button_frame,
    text="Clear",
    font=("Arial", 12),
    command=clear_canvas
)
clear_button.pack(side=tk.LEFT, padx=10)

predict_button = tk.Button(
    button_frame,
    text="Predict Digit",
    font=("Arial", 12, "bold"),
    command=predict_digit
)
predict_button.pack(side=tk.LEFT, padx=10)

result_label = tk.Label(
    root,
    text="Prediction: ?",
    font=("Arial", 20, "bold"),
    bg="#1e1e2e",
    fg="#a6e3a1"
)
result_label.pack(pady=5)

probabilities_label = tk.Label(
    root,
    text="",
    font=("Consolas", 11),
    justify=tk.LEFT,
    bg="#1e1e2e",
    fg="white"
)
probabilities_label.pack(pady=5)

root.mainloop()
