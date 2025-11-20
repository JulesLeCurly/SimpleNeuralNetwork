import numpy as np

def min_max_scaling(data):
    # If all values are the same, return a constant value (0.5)
    """
    This function scales the input data to the range [0, 1].
    """
    min_data = np.min(data)
    max_data = np.max(data)

    scaled_data = (data - min_data) / (max_data - min_data)

    return scaled_data