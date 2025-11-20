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
        epochs=1,
        learning_rate=0.1,
        shuffle=True,
        verbose=False
    ):
        """Trains the model for a fixed number of epochs (iterations on a dataset).

        Args:
            x: Input data. It could be:
              - A Numpy array (or array-like), or a list of arrays
                (in case the model has multiple inputs).
              - A TensorFlow tensor, or a list of tensors
                (in case the model has multiple inputs).
              - A dict mapping input names to the corresponding array/tensors,
                if the model has named inputs.
              - A `tf.data` dataset. Should return a tuple
                of either `(inputs, targets)` or
                `(inputs, targets, sample_weights)`.
              - A generator or `keras.utils.Sequence` returning `(inputs,
                targets)` or `(inputs, targets, sample_weights)`.
              - A `tf.keras.utils.experimental.DatasetCreator`, which wraps a
                callable that takes a single argument of type
                `tf.distribute.InputContext`, and returns a `tf.data.Dataset`.
                `DatasetCreator` should be used when users prefer to specify the
                per-replica batching and sharding logic for the `Dataset`.
                See `tf.keras.utils.experimental.DatasetCreator` doc for more
                information.
              A more detailed description of unpacking behavior for iterator
              types (Dataset, generator, Sequence) is given below. If these
              include `sample_weights` as a third component, note that sample
              weighting applies to the `weighted_metrics` argument but not the
              `metrics` argument in `compile()`. If using
              `tf.distribute.experimental.ParameterServerStrategy`, only
              `DatasetCreator` type is supported for `x`.
            y: Target data. Like the input data `x`,
              it could be either Numpy array(s) or TensorFlow tensor(s).
              It should be consistent with `x` (you cannot have Numpy inputs and
              tensor targets, or inversely). If `x` is a dataset, generator,
              or `keras.utils.Sequence` instance, `y` should
              not be specified (since targets will be obtained from `x`).
            batch_size: Integer or `None`.
                Number of samples per gradient update.
                If unspecified, `batch_size` will default to 32.
                Do not specify the `batch_size` if your data is in the
                form of datasets, generators, or `keras.utils.Sequence`
                instances (since they generate batches).
            epochs: Integer. Number of epochs to train the model.
                An epoch is an iteration over the entire `x` and `y`
                data provided
                (unless the `steps_per_epoch` flag is set to
                something other than None).
                Note that in conjunction with `initial_epoch`,
                `epochs` is to be understood as "final epoch".
                The model is not trained for a number of iterations
                given by `epochs`, but merely until the epoch
                of index `epochs` is reached.
            verbose: 'auto', 0, 1, or 2. Verbosity mode.
                0 = silent, 1 = progress bar, 2 = one line per epoch.
                'auto' defaults to 1 for most cases, but 2 when used with
                `ParameterServerStrategy`. Note that the progress bar is not
                particularly useful when logged to a file, so verbose=2 is
                recommended when not running interactively (eg, in a production
                environment).
            callbacks: List of `keras.callbacks.Callback` instances.
                List of callbacks to apply during training.
                See `tf.keras.callbacks`. Note
                `tf.keras.callbacks.ProgbarLogger` and
                `tf.keras.callbacks.History` callbacks are created automatically
                and need not be passed into `model.fit`.
                `tf.keras.callbacks.ProgbarLogger` is created or not based on
                `verbose` argument to `model.fit`.
                Callbacks with batch-level calls are currently unsupported with
                `tf.distribute.experimental.ParameterServerStrategy`, and users
                are advised to implement epoch-level calls instead with an
                appropriate `steps_per_epoch` value.
            validation_split: Float between 0 and 1.
                Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
                The validation data is selected from the last samples
                in the `x` and `y` data provided, before shuffling. This
                argument is not supported when `x` is a dataset, generator or
                `keras.utils.Sequence` instance.
                If both `validation_data` and `validation_split` are provided,
                `validation_data` will override `validation_split`.
                `validation_split` is not yet supported with
                `tf.distribute.experimental.ParameterServerStrategy`.
            validation_data: Data on which to evaluate
                the loss and any model metrics at the end of each epoch.
                The model will not be trained on this data. Thus, note the fact
                that the validation loss of data provided using
                `validation_split` or `validation_data` is not affected by
                regularization layers like noise and dropout.
                `validation_data` will override `validation_split`.
                `validation_data` could be:
                  - A tuple `(x_val, y_val)` of Numpy arrays or tensors.
                  - A tuple `(x_val, y_val, val_sample_weights)` of NumPy
                    arrays.
                  - A `tf.data.Dataset`.
                  - A Python generator or `keras.utils.Sequence` returning
                  `(inputs, targets)` or `(inputs, targets, sample_weights)`.
                `validation_data` is not yet supported with
                `tf.distribute.experimental.ParameterServerStrategy`.
            shuffle: Boolean (whether to shuffle the training data
                before each epoch) or str (for 'batch'). This argument is
                ignored when `x` is a generator or an object of tf.data.Dataset.
                'batch' is a special option for dealing
                with the limitations of HDF5 data; it shuffles in batch-sized
                chunks. Has no effect when `steps_per_epoch` is not `None`.
            class_weight: Optional dictionary mapping class indices (integers)
                to a weight (float) value, used for weighting the loss function
                (during training only).
                This can be useful to tell the model to
                "pay more attention" to samples from
                an under-represented class.
            sample_weight: Optional Numpy array of weights for
                the training samples, used for weighting the loss function
                (during training only). You can either pass a flat (1D)
                Numpy array with the same length as the input samples
                (1:1 mapping between weights and samples),
                or in the case of temporal data,
                you can pass a 2D array with shape
                `(samples, sequence_length)`,
                to apply a different weight to every timestep of every sample.
                This argument is not supported when `x` is a dataset, generator,
                or `keras.utils.Sequence` instance, instead provide the
                sample_weights as the third element of `x`.
                Note that sample weighting does not apply to metrics specified
                via the `metrics` argument in `compile()`. To apply sample
                weighting to your metrics, you can specify them via the
                `weighted_metrics` in `compile()` instead.
            initial_epoch: Integer.
                Epoch at which to start training
                (useful for resuming a previous training run).
            steps_per_epoch: Integer or `None`.
                Total number of steps (batches of samples)
                before declaring one epoch finished and starting the
                next epoch. When training with input tensors such as
                TensorFlow data tensors, the default `None` is equal to
                the number of samples in your dataset divided by
                the batch size, or 1 if that cannot be determined. If x is a
                `tf.data` dataset, and 'steps_per_epoch'
                is None, the epoch will run until the input dataset is
                exhausted.  When passing an infinitely repeating dataset, you
                must specify the `steps_per_epoch` argument. If
                `steps_per_epoch=-1` the training will run indefinitely with an
                infinitely repeating dataset.  This argument is not supported
                with array inputs.
                When using `tf.distribute.experimental.ParameterServerStrategy`:
                  * `steps_per_epoch=None` is not supported.
            validation_steps: Only relevant if `validation_data` is provided and
                is a `tf.data` dataset. Total number of steps (batches of
                samples) to draw before stopping when performing validation
                at the end of every epoch. If 'validation_steps' is None,
                validation will run until the `validation_data` dataset is
                exhausted. In the case of an infinitely repeated dataset, it
                will run into an infinite loop. If 'validation_steps' is
                specified and only part of the dataset will be consumed, the
                evaluation will start from the beginning of the dataset at each
                epoch. This ensures that the same validation samples are used
                every time.
            validation_batch_size: Integer or `None`.
                Number of samples per validation batch.
                If unspecified, will default to `batch_size`.
                Do not specify the `validation_batch_size` if your data is in
                the form of datasets, generators, or `keras.utils.Sequence`
                instances (since they generate batches).
            validation_freq: Only relevant if validation data is provided.
              Integer or `collections.abc.Container` instance (e.g. list, tuple,
              etc.).  If an integer, specifies how many training epochs to run
              before a new validation run is performed, e.g. `validation_freq=2`
              runs validation every 2 epochs. If a Container, specifies the
              epochs on which to run validation, e.g.
              `validation_freq=[1, 2, 10]` runs validation at the end of the
              1st, 2nd, and 10th epochs.
            max_queue_size: Integer. Used for generator or
              `keras.utils.Sequence` input only. Maximum size for the generator
              queue.  If unspecified, `max_queue_size` will default to 10.
            workers: Integer. Used for generator or `keras.utils.Sequence` input
                only. Maximum number of processes to spin up
                when using process-based threading. If unspecified, `workers`
                will default to 1.
            use_multiprocessing: Boolean. Used for generator or
                `keras.utils.Sequence` input only. If `True`, use process-based
                threading. If unspecified, `use_multiprocessing` will default to
                `False`. Note that because this implementation relies on
                multiprocessing, you should not pass non-picklable arguments to
                the generator as they can't be passed easily to children
                processes.

        Unpacking behavior for iterator-like inputs:
            A common pattern is to pass a tf.data.Dataset, generator, or
          tf.keras.utils.Sequence to the `x` argument of fit, which will in fact
          yield not only features (x) but optionally targets (y) and sample
          weights.  Keras requires that the output of such iterator-likes be
          unambiguous. The iterator should return a tuple of length 1, 2, or 3,
          where the optional second and third elements will be used for y and
          sample_weight respectively. Any other type provided will be wrapped in
          a length one tuple, effectively treating everything as 'x'. When
          yielding dicts, they should still adhere to the top-level tuple
          structure.
          e.g. `({"x0": x0, "x1": x1}, y)`. Keras will not attempt to separate
          features, targets, and weights from the keys of a single dict.
            A notable unsupported data type is the namedtuple. The reason is
          that it behaves like both an ordered datatype (tuple) and a mapping
          datatype (dict). So given a namedtuple of the form:
              `namedtuple("example_tuple", ["y", "x"])`
          it is ambiguous whether to reverse the order of the elements when
          interpreting the value. Even worse is a tuple of the form:
              `namedtuple("other_tuple", ["x", "y", "z"])`
          where it is unclear if the tuple was intended to be unpacked into x,
          y, and sample_weight or passed through as a single element to `x`. As
          a result the data processing code will simply raise a ValueError if it
          encounters a namedtuple. (Along with instructions to remedy the
          issue.)

        Returns:
            A `History` object. Its `History.history` attribute is
            a record of training loss values and metrics values
            at successive epochs, as well as validation loss values
            and validation metrics values (if applicable).

        Raises:
            RuntimeError: 1. If the model was never compiled or,
            2. If `model.fit` is  wrapped in `tf.function`.

            ValueError: In case of mismatch between the provided input data
                and what the model expects or when the input data is empty.
        """
        # Check if n_iter is a positive integer
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("The number of iterations should be a positive integer.")

        # Check if learning_rate is a positive number
        if (not isinstance(learning_rate, (int, float)) or learning_rate <= 0) and learning_rate != "Auto":
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
        training_history = np.zeros((int(epochs), 2))

        # Gradient descent training loop
        if verbose:
            for i in tqdm(range(epochs)):
                training_history = self.learning_process(i, X, Y, learning_rate, training_history)
        else:
            for i in range(epochs):
                training_history = self.learning_process(i, X, Y, learning_rate, training_history)

        return training_history 
    
    def learning_process(
        self,
        i,
        X,
        y,
        learning_rate,
        training_history
    ):
        # Forward propagation
        self.forward_propagation(X)
        # Backward propagation
        self.back_propagation(y)
        # Update weights and biases
        self.update(learning_rate)
        Af = self.activations['A' + str(self.C)]

        # Compute log loss
        training_history[i, 0] = log_loss(y.flatten(), Af.flatten())

        # Compute accuracy
        y_pred = self.BRUTE_predict(X)
        training_history[i, 1] = accuracy_score(y.flatten(), np.round(y_pred).flatten())
        return training_history