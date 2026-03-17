import numpy as np

def quadratic_loss(s, Y):
    return (1/2)*np.sum((s - Y)**2)

def quadratic_loss_der(s, Y):
    return (s - Y)
    