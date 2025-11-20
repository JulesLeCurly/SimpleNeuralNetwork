import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def relu(x):
    return np.maximum(0, x)

def derivative_relu(x):
    return np.where(x > 0, 1, 0)

def leaky_relu(x, alpha=0.1):
    return np.maximum(alpha * x, x)

def derivative_leaky_relu(x, alpha=0.1):
    return np.where(x > 0, 1, alpha)

def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=0)

def derivative_softmax(x):
    return softmax(x) * (1 - softmax(x))

def tanh(x):
    return np.tanh(x)

def derivative_tanh(x):
    return 1 - np.tanh(x)**2