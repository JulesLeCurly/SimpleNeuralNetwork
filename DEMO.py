import SimpleNeuralNetwork as SNN
import numpy as np

# Define input (X) and output (y) data
X = np.array([[3, 1.5], [2, 1], [4, 1.5], [3, 1], 
              [3.5, 0.5], [2, 0.5], [5.5, 1], [1, 1]], dtype=float)
Y = np.array([[1], [0], [1], [0], [1], [0], [1], [0]], dtype=float)

#Standardization (not obligate) of the data if they are not a float between 0 and 1
X = SNN.standardization.min_max_scaling(X)

# Build the model
model = SNN.model(
    [
        SNN.layers.InputLayer(2),
        SNN.layers.Dense(3, activation="sigmoid"),
        SNN.layers.Dense(1, activation="sigmoid"),
    ]
)

# Train the model
model.train(X, Y, epochs=100, learning_rate="Auto", shuffle=True, verbose=True)

# Save the model
model.save("model_save_test")

# Load the model
model_loaded = SNN.load("model_save_test")

# Make prediction
a = model_loaded.predict(np.array([[4, 1.5]]))
print(f"    ~~Prediction of the model~~    ")
print(a)

# Info
model_loaded.summary()