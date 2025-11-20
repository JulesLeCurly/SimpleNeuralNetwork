import SimpleNeuralNetwork as SNN
import numpy as np

# Define input (X) and output (y) data
X = np.array([[3, 1.5], [2, 1], [4, 1.5], [3, 1], 
              [3.5, 0.5], [2, 0.5], [5.5, 1], [1, 1]], dtype=np.float16)
Y = np.array([[1], [0], [1], [0], [1], [0], [1], [0]], dtype=np.float16)

# Exemple : dataset XOR
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

Y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)

# Normalize the input
X_max = np.max(X, axis=0)
X /= X_max

# Build the model
model = SNN.model(
    [
        SNN.layers.InputLayer(2),
        SNN.layers.Dense(3, activation="sigmoid"),
        SNN.layers.Dense(1, activation="sigmoid"),
    ]
)

# Train the model
model.train(X, Y, epochs=100, learning_rate=0.001, shuffle=True, verbose=True)

# Save the model
#SNN.save(model, "model_save_test") # Not working yet

# Load the model
#model_loaded = SNN.load("model_save_test") # Not working yet
model_loaded = model # Because of not Save and Load available

# Make prediction
a = model_loaded.predict(np.array([[1, 0]]) / X_max)
print(f"    ~~Prediction of the model~~    ")
print(a)

# Info
#model_loaded.summary() # Not working yet