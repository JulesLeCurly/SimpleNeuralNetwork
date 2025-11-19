# SimpleNeuralNetwork

# Warning : Not fonctionnale project yet. (sorry)

A lightweight Python library for building simple neural networks with bias, hidden layers, and classic neurons, featuring fast and efficient CPU-based prediction using NumPy.

![License](https://img.shields.io/badge/license-MIT-green)

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Tests](#tests)
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
from simple_neural_network import SimpleNeuralNetwork

# Example: creating a neural network with 1 input layer, 1 hidden layer, and 1 output layer
nn = SimpleNeuralNetwork(input_size=2, hidden_size=3, output_size=1)

# Training data (X) and labels (y)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Example inputs (X)
y = np.array([[0], [1], [1], [0]])  # Example outputs (y)

# Train the neural network
nn.train(X, y, epochs=1000, learning_rate=0.1)

# Make predictions
predictions = nn.predict(X)
print(predictions)
```

## Tests

To run the tests for the SimpleNeuralNetwork library, make sure you have `pytest` installed:

```bash
pip install pytest
```

Then, run the tests with:

```bash
pytest
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

- **Ton Nom** - *Initial work* - [JulesLeCurly](https://github.com/JulesLeCurly)
```