import numpy as np

def leaky_relu(x, alpha=0.01):
    """
    Vectorized Leaky ReLU implementation.
    """
    x = np.array(x)   # convert to numpy array
    return np.maximum(alpha * x, x)