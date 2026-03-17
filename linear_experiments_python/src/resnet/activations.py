import numpy as np

def tanh(s):
    return np.tanh(s)

def tanh_der(s):
    return 1.0 / (np.cosh(s) ** 2)

def linear(s):
    return s

def linear_der(s):
    return np.ones_like(s)
    