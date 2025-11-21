import SimpleNeuralNetwork as SNN
import numpy as np
import matplotlib.pyplot as plt

# Define input (X) and output (y) data
X = np.array([[3, 1.5], [2, 1], [4, 1.5], [3, 1], 
              [3.5, 0.5], [2, 0.5], [5.5, 1], [1, 1]], dtype=np.float16)
Y = np.array([[1], [0], [1], [0], [1], [0], [1], [0]], dtype=np.float16)

# Normalize the input
X_max = np.max(X, axis=0)
X /= X_max

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
history = model.train(X, Y, epochs=400, learning_rate="Auto", shuffle=True, verbose=True)
history = history.T
print(history[0][400 - 1])

# Plot the loss
plt.plot(history[0])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Save the model
#SNN.save(model, "model_save_test") # Not working yet

# Load the model
#model_loaded = SNN.load("model_save_test") # Not working yet
model_loaded = model # Because of not Save and Load available

# Make prediction
pred = model_loaded.predict(np.array([[4, 1.5]]) / X_max)
print(f"    ~~Prediction of the model~~    ")
print(pred)

# Info
#model_loaded.summary() # Not working yet