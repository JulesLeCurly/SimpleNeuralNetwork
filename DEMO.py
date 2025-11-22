import SimpleNeuralNetwork as SNN
import numpy as np
import matplotlib.pyplot as plt
import time

import Demo.Generate_dataset

# Define input (X) and output (y) data
X, Y = Demo.Generate_dataset.generate_circle_dataset(num_samples=100, noise=0.05)
X_test, Y_test = Demo.Generate_dataset.generate_circle_dataset(num_samples=50, noise=0.05)

# Build the model
model = SNN.model(
    [
        SNN.layers.InputLayer(2),
        SNN.layers.Dense(3, activation="sigmoid"),
        SNN.layers.Dense(1, activation="sigmoid"),
    ],
    seed=42
)

# Train the model
Start = time.time()
history = model.train(X, Y, X_test, Y_test, epochs=3000, learning_rate="Auto", shuffle=True, verbose=True)
End = time.time()
print(f"Training time: {End - Start:.2f} seconds")

# Plot the loss
plt.plot(history["loss"], label="Training Loss", color="blue")
plt.plot(history["test_loss"], label="Test Loss", color="orange")
plt.title("Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Save the model
#SNN.save(model, "model_save_test") # Not working yet

# Load the model
#model_loaded = SNN.load("model_save_test") # Not working yet
model_loaded = model # Because of not Save and Load available

# Make prediction
pred = model_loaded.predict(np.array([X_test[0]]))
print(f"    ~~Prediction of the model~~    ")
print(f"For the first sample: {X_test[0]} -> predicted label: {pred[0]:.2f} expected label: {Y_test[0]}")

# Info
#model_loaded.summary() # Not working yet