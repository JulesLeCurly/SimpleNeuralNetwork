# SimpleNeuralNetwork

# Warning : Not fonctionnale project yet. (sorry)

A lightweight Python library for building simple neural networks with bias, hidden layers, and classic neurons, featuring fast and efficient CPU-based prediction using NumPy.

![License](https://img.shields.io/badge/license-MIT-green)

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Authors](#authors)

## Installation

To install the SimpleNeuralNetwork library, run:m

```bash
pip install https://github.com/JulesLeCurly/SimpleNeuralNetwork.git
```

## Usage

Here is a simple example of how to use the library to create and train a basic neural network:

```python
import SimpleNeuralNetwork as SNN
import numpy as np

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
    ]
)

# Train the model
history = model.train(X, Y, epochs=400, learning_rate=1, shuffle=True, verbose=True)

# Make prediction
pred = model_loaded.predict(np.array([[4, 1.5]]) / X_max)
print(f"Prediction of the model : {pred}")
```

## Contributing

We welcome contributions! If you'd like to contribute, please fork the repository, make your changes, and submit a pull request. Here are a few ways you can help:

- Report bugs
- Suggest new features
- Improve documentation
- Fix issues or add tests

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Jules* - *Initial work* - [JulesLeCurly](https://github.com/JulesLeCurly)
```