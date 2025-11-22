from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm
import numpy as np
import random
import os

class model():
    def __init__(
        self,
        NeuralNetwork_structure,
        Patametres_exist = None,
        seed=None
    ):
        """
        Initialize a neural network with specified layer dimensions and random seed.
        
                Parameters:
                -----------
                NeuralNetwork_structure : list
                    A list of integers specifying the number of neurons in each layer.
                    The first element represents the input layer size,
                    The last element represents the output layer size,
                    And elements in between represent hidden layer sizes.
                
                seed : int, optional
                    Random seed for weight initialization to ensure reproducibility.
        
        """
        # Check if the input 'dimensions' is a list
        if not isinstance(NeuralNetwork_structure, list) and Patametres_exist is None:
            raise TypeError("""
NeuralNetwork_structure should be a list. Example:
[
SimpleNeuralNetwork.layers.InputLayer(2),
SimpleNeuralNetwork.layers.Dense(3, activation="sigmoid"),
SimpleNeuralNetwork.layers.Dense(1, activation="sigmoid"),
]
""")
        # Check if the input 'dimensions' is not empty
        if (not NeuralNetwork_structure or NeuralNetwork_structure == []) and Patametres_exist is None:
            raise ValueError("NeuralNetwork_structure list cannot be empty.")
        
        # Check if seed is an integer
        if not isinstance(seed, int) and not seed is None:
            raise TypeError("The seed should be an integer.")

        # Initialize random seed for reproducibility
        if not seed is None:
            np.random.seed(seed)

        # Dictionary to store parameters (weights and biases)
        if Patametres_exist is not None:
            self.parametres = Patametres_exist
            self.nb_layers = len([key for key in self.parametres.keys() if 'W' in key])

        else:
            self.parametres = {}

            # Total number of layers
            self.nb_layers = len(NeuralNetwork_structure) - 1

            # Store the dimensions of each layer
            self.NeuralNetwork_structure = NeuralNetwork_structure
            
            print(self.nb_layers)
            # Initialize weights and biases for each layer (starting from layer 1)
            for c in range(1, self.nb_layers + 1):
                self.parametres['W' + str(c)] = np.random.randn(NeuralNetwork_structure[c].units, NeuralNetwork_structure[c - 1].units)
                self.parametres['b' + str(c)] = np.random.randn(NeuralNetwork_structure[c].units, 1)
        
        print(self.parametres)
        print(self.NeuralNetwork_structure)

        # Update C to represent the number of layers with parameters (excluding input layer)
        self.C = len(self.parametres) // 2

        # Other attributes
        self.learning_rate = None
        self.Auto_learning_rate = None

    def forward_propagation(
        self,
        X
    ):
        # Store input data as activation of layer 0
        self.activations = {'A0': X}

        # Forward pass through each layer
        for c in range(1, self.C + 1):
            # Linear combination: Z = W * A_prev + b
            Z = self.parametres['W' + str(c)].dot(self.activations['A' + str(c - 1)]) + self.parametres['b' + str(c)]
            # Activation using sigmoid function
            self.activations['A' + str(c)] = 1 / (1 + np.exp(-Z))  # Activation

    def back_propagation(
        self,
        Y
    ):
        # Number of training examples
        m = Y.shape[1]

        # Dictionary to store gradients for weights and biases
        self.gradients = {}

        # Compute the derivative of the loss with respect to the output activation
        dZ = self.activations['A' + str(self.C)] - Y

        # Backward pass through each layer (from last to first)
        for c in reversed(range(1, self.C + 1)):
            # Gradient of the loss with respect to weights
            self.gradients['dW' + str(c)] = 1/m * np.dot(dZ, self.activations['A' + str(c - 1)].T)
            # Gradient of the loss with respect to biases
            self.gradients['db' + str(c)] = 1/m * np.sum(dZ, axis=1, keepdims=True)
            
            # Compute dZ for the previous layer if not at the first layer
            if c > 1:
                # Chain rule: propagate error backwards using the derivative of the sigmoid activation
                dZ = np.dot(self.parametres['W' + str(c)].T, dZ) * self.activations['A' + str(c - 1)] * (1 - self.activations['A' + str(c - 1)])



    def update(
        self,
        learning_rate
    ):
        # Update weights and biases using gradient descent
        for c in range(1, self.C + 1):
            # Update weights: W = W - learning_rate * dW
            self.parametres['W' + str(c)] = self.parametres['W' + str(c)] - learning_rate * self.gradients['dW' + str(c)]
            # Update biases: b = b - learning_rate * db
            self.parametres['b' + str(c)] = self.parametres['b' + str(c)] - learning_rate * self.gradients['db' + str(c)]


    def BRUTE_predict(
        self,
        X
    ):
        # Perform forward propagation to get final output
        self.forward_propagation(X)
        # Return the activation of the final layer (output)
        Af = self.activations['A' + str(self.C)]
        return Af


    def predict(
        self,
        input_data
    ):
        # Check that input_data is a list
        if not isinstance(input_data, np.ndarray):
            raise TypeError("The function's input data must be a numpy array.")
        
        # Validate input dimensions
        expected_input_dim = self.NeuralNetwork_structure[0].units

        if input_data.shape[1] != expected_input_dim:
            raise ValueError(
                f"Invalid shape for X: expected (?, {expected_input_dim}), got {X.shape}"
            )
        
        # Convert input data to numpy array
        X = np.array(input_data, dtype=float)
        X = X.T
        X = X.reshape(X.shape[0], -1)

        # Get prediction from the model
        prediction = self.BRUTE_predict(X).flatten()
        return prediction

    def train(
        self,
        X=None,
        Y=None,
        X_test=None,
        Y_test=None,
        epochs=1,
        learning_rate=0.1,
        shuffle=True,
        verbose=False
    ):
        # Store the learning rate
        self.learning_rate = learning_rate
        if self.learning_rate == "Auto":
            self.Auto_learning_rate = True
        else:
            self.Auto_learning_rate = False

        # Check if n_iter is a positive integer
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("The number of iterations should be a positive integer.")

        # Check if learning_rate is a positive number
        if (not isinstance(self.learning_rate, (int, float)) or self.learning_rate <= 0) and self.learning_rate != "Auto":
            raise ValueError("The learning_rate should be a positive number or 'Auto'.")

        # Check if show is a boolean
        if not isinstance(verbose, bool):
            raise ValueError("The 'show' parameter should be a boolean value.")

        # Check that all output elements are lists
        if any(not isinstance(val, np.ndarray) for val in Y):
            raise ValueError("The output numpy array must only contain array.")

        # Check that all input elements are lists
        if any(not isinstance(val, np.ndarray) for val in X):
            raise ValueError("The input numpy array must only contain array.")

        # Validate output dimensions
        expected_output_dim = self.NeuralNetwork_structure[self.nb_layers].units

        if Y.ndim != 2 or Y.shape[1] != expected_output_dim:
            raise ValueError(
                f"Invalid shape for Y: expected (n, {expected_output_dim}), got {Y.shape}"
            )
        
        # Validate input dimensions
        X = np.asarray(X, dtype=float)
        expected_input_dim = self.NeuralNetwork_structure[0].units

        if X.ndim != 2 or X.shape[1] != expected_input_dim:
            raise ValueError(
                f"Invalid shape for X: expected (?, {expected_input_dim}), got {X.shape}"
            )

        # Transpose Y
        Y = np.transpose(Y)
        X = X.T
        X = X.reshape(X.shape[0], -1)

        # Array to store training loss and accuracy for each iteration
        training_history = {
            'loss': np.zeros((epochs)),
            'accuracy': np.zeros((epochs)),
            'test_loss': np.zeros((epochs)),
            'test_accuracy': np.zeros((epochs))
        }

        if X_test is None and Y_test is None:
            training_history.pop('test_loss')
            training_history.pop('test_accuracy')

        # Set learning rate
        if self.Auto_learning_rate:
            self.Set_learning_rate(-1, training_history)
        
        # Gradient descent training loop
        if verbose:
            for i in tqdm(range(epochs)):
                training_history = self.learning_process(i, X, Y, X_test, Y_test, training_history)
        else:
            for i in range(epochs):
                training_history = self.learning_process(i, X, Y, X_test, Y_test, training_history)

        return training_history 
    
    def learning_process(
        self,
        i,
        X,
        y,
        X_test,
        Y_test,
        training_history
    ):
        # Forward propagation
        self.forward_propagation(X)
        # Backward propagation
        self.back_propagation(y)
        # Update weights and biases
        self.update(self.learning_rate)
        Af = self.activations['A' + str(self.C)]

        # Compute log loss
        training_history["loss"][i] = log_loss(y.flatten(), Af.flatten())
        if not X_test is None and not Y_test is None:
            training_history["test_loss"][i] = log_loss(Y_test.flatten(), self.predict(X_test).flatten())

        # Compute accuracy
        y_pred = self.BRUTE_predict(X)
        training_history["accuracy"][i] = accuracy_score(y.flatten(), np.round(y_pred).flatten())
        if not X_test is None and not Y_test is None:
            training_history["test_accuracy"][i] = accuracy_score(Y_test.flatten(), np.round(self.predict(X_test)).flatten())

        # Set learning rate
        if self.Auto_learning_rate:
            self.Set_learning_rate(i, training_history)
            
        return training_history
    
    def Set_learning_rate(self, iteration, loss_history):
        if iteration == -1:
            self.learning_rate = 2
        else:
            if loss_history["loss"][iteration] > loss_history["loss"][iteration - 1]:
                self.learning_rate *= 0.9