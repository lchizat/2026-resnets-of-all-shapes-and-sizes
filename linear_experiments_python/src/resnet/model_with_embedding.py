import numpy as np

from src.resnet.model import FiniteResNet
from .activations import tanh, tanh_der
from .losses import quadratic_loss, quadratic_loss_der

class FiniteResNetWithEmbedding(FiniteResNet):
    def __init__(self, D, M, L, d_in, d_out, alpha=1.0, activation=tanh, activation_der=tanh_der, loss=quadratic_loss, loss_der=quadratic_loss_der, seed = 0):
        np.random.seed(seed)
        super().__init__(D, M, L, alpha, activation, activation_der, loss, loss_der)
        self.W_e = np.random.randn(D, d_in) # assuming fixed variance 1 
        self.W_u = np.random.randn(D, d_out) # assuming fixed variance 1

    def forward(self, X):
        d_in, N = X.shape
        E = self.W_e @ X
        
        H, P = super().forward(E)
        
        out = (1/self.D)*(self.W_u.T @ H[:, :, self.L])

        return out, H, P

    def step(self, X, Y, eta, eta_u=1.0, eta_v=1.0, track=False):
        # 1. Forward Pass
        out, H, P = self.forward(X)
        D, N = H[:, :, self.L].shape

        # 2. Backward Pass (Compute B)
        # Init condition: b(final) = gradient of loss w.r.t. final output
        W = (1/D) * self.W_u @ self.loss_der(out, Y)
        B, Q = self.backward(W, H, P)

        # 3. Compute gradients
        gradU, gradV = self.compute_gradients(P, H, B, Q)

        # 4. Apply Update (using -= as U and V are instance variables)
        self.U -= eta_u * (eta / self.alpha) * (1/N) * gradU
        self.V -= eta_v * (eta / self.alpha) * (1/N) * gradV
        
        current_loss = (1/N)*self.loss(out, Y)
        if track:
            return current_loss, H, B, out
        return current_loss