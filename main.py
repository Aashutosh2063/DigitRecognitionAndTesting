import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

print("Loading MNIST...")

mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False
)

X = mnist.data
y = mnist.target.astype(np.uint8)

print("Total images:", X.shape[0])
print("Pixels per image:", X.shape[1])

example_index = 10

image = X[example_index].reshape(28, 28)
label = y[example_index]

print("\n--- Example Image ---")
print("Image index:", example_index)
print("Actual digit:", label)
print("Image data type:", type(X))

plt.imshow(image, cmap="gray")
plt.title(f"Actual digit: {label} (Image #{example_index})")
plt.axis("on")
plt.show()

input_vector = image.flatten()
input_vector = input_vector / 255.0

print("Image shape:", image.shape)
print("Input vector shape:", input_vector.shape)
print("First 20 pixel values:", input_vector[:20])

input_size = 784
hidden_size = 128
output_size = 10

weights1 = np.random.randn(hidden_size, input_size) * 0.01
biases1 = np.zeros(hidden_size)

weights2 = np.random.randn(output_size, hidden_size) * 0.01
biases2 = np.zeros(output_size)

print("\n--- Network Structure ---")
print("Input layer:", input_size, "pixels")
print("Hidden layer:", hidden_size, "neurons")
print("Output layer:", output_size, "neurons")
print("First weight matrix:", weights1.shape)
print("Second weight matrix:", weights2.shape)

z1 = np.dot(weights1, input_vector) + biases1
a1 = np.maximum(0, z1)

z2 = np.dot(weights2, a1) + biases2

print("\n--- Forward Pass ---")
print("Hidden layer output shape:", a1.shape)
print("Output scores:", z2)
print("Output shape:", z2.shape)

exp_scores = np.exp(z2 - np.max(z2))
probabilities = exp_scores / np.sum(exp_scores)

predicted_digit = np.argmax(probabilities)

print("\n--- Before Training ---")
print("Image used:", example_index)
print("Actual digit:", label)
print("Predicted digit:", predicted_digit)
print("Probabilities:", np.round(probabilities, 4))

loss = -np.log(probabilities[label] + 1e-9)
print("Loss:", round(float(loss), 4))

true_labels = np.zeros(10)
true_labels[label] = 1

dz2 = probabilities - true_labels

dw2 = np.outer(dz2, a1)
db2 = dz2

da1 = np.dot(weights2.T, dz2)
dz1 = da1 * (z1 > 0)

dw1 = np.outer(dz1, input_vector)
db1 = dz1

learning_rate = 0.01

weights2 -= learning_rate * dw2
biases2 -= learning_rate * db2

weights1 -= learning_rate * dw1
biases1 -= learning_rate * db1

z1 = np.dot(weights1, input_vector) + biases1
a1 = np.maximum(0, z1)

z2 = np.dot(weights2, a1) + biases2

exp_scores = np.exp(z2 - np.max(z2))
probabilities = exp_scores / np.sum(exp_scores)

predicted_digit = np.argmax(probabilities)

print("\n--- After One Training Step ---")
print("Image used:", example_index)
print("Actual digit:", label)
print("Predicted digit:", predicted_digit)
print("Probabilities:", np.round(probabilities, 4))
print("Loss:", round(float(-np.log(probabilities[label] + 1e-9)), 4))

training_size = 20000

X_train = X[:training_size].astype(np.float32) / 255.0
y_train = y[:training_size]

print("\n--- Training Data ---")
print("First training image: X[2]")
print("Last training image: X[999]")
print("Number of training images:", len(X_train))
print("Training image shape:", X_train[0].shape)
print("Training labels shape:", y_train.shape)

learning_rate = 0.01
epochs = 10

for epoch in range(epochs):
    total_loss = 0
    correct = 0

    shuffled_indices = np.random.permutation(len(X_train))

    for i in shuffled_indices:

        x = X_train[i]
        label = y_train[i]

        # Forward pass
        z1 = weights1 @ x + biases1
        a1 = np.maximum(0, z1)

        z2 = weights2 @ a1 + biases2
        p = np.exp(z2 - np.max(z2))
        p = p / np.sum(p)

        total_loss += -np.log(p[label] + 1e-9)
        correct += int(np.argmax(p) == label)

        target = np.zeros(10)
        target[label] = 1

        dz2 = p - target
        dw2 = np.outer(dz2, a1)
        db2 = dz2

        dz1 = (weights2.T @ dz2) * (z1 > 0)
        dw1 = np.outer(dz1, x)
        db1 = dz1

        weights2 -= learning_rate * dw2
        biases2 -= learning_rate * db2
        weights1 -= learning_rate * dw1
        biases1 -= learning_rate * db1

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Images processed: {training_size} | "
        f"Average loss: {total_loss / training_size:.4f} | "
        f"Training accuracy: {correct / training_size * 100:.2f}%"
    )

X_test = X[60000:61000].astype(np.float32) / 255.0
y_test = y[60000:61000]

correct = 0

for i in range(len(X_test)):
    x = X_test[i]
    label = y_test[i]

    z1 = weights1 @ x + biases1
    a1 = np.maximum(0, z1)

    z2 = weights2 @ a1 + biases2

    exp_scores = np.exp(z2 - np.max(z2))
    probabilities = exp_scores / np.sum(exp_scores)

    predicted_digit = np.argmax(probabilities)

    if predicted_digit == label:
        correct += 1

accuracy = correct / len(X_test) * 100
print("\n--- Test Results ---")
print("Test images:", len(X_test))
print("Correct predictions:", correct)
print(f"Test accuracy: {accuracy:.2f}%")

np.savez(
    "trained_network.npz",
    weights1=weights1,
    biases1=biases1,
    weights2=weights2,
    biases2=biases2
)

print("Trained network saved successfully!")
