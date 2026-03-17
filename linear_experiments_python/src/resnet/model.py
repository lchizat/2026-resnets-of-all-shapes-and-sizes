import numpy as np
from .activations import tanh, tanh_der
from .losses import quadratic_loss, quadratic_loss_der

class FiniteResNet:
    def __init__(self, D, M, L, alpha=1.0, activation=tanh, activation_der=tanh_der, loss=quadratic_loss, loss_der=quadratic_loss_der):
        self.D = D
        self.M = M
        self.L = L
        self.alpha = alpha
        self.sigma = activation
        self.sigma_der = activation_der
        self.loss = loss
        self.loss_der = loss_der

        sigma = np.sqrt(D)
        self.U = np.random.randn(D, M, L) * sigma      # shape (D, M, L)
        self.V = np.random.randn(D, M, L) * sigma

    def forward(self, X):
        D, N = X.shape
        H = np.zeros((D, N, self.L + 1))
        P = np.zeros((self.M, N, self.L))

        H[:, :, 0] = X

        for l in range(self.L):
            P[:, :, l] = (self.U[:, :, l].T @ H[:, :, l]) / D
            H[:, :, l + 1] = H[:, :, l] + (self.alpha / (self.L * self.M)) * (self.V[:, :, l] @ self.sigma(P[:, :, l]))

        return H, P

    def backward(self, W, H, P):
        B = np.zeros_like(H)
        Q = np.zeros((self.M, B.shape[1], self.L))
        # Init condition: b(final) = gradient of loss w.r.t. final output
        B[:, :, self.L] = W

        for l in range(self.L - 1, -1, -1):
            Q[:, :, l ] = self.sigma_der(P[:, :, l]) * (self.V[:, :, l].T @ B[:, :, l + 1])
            B[:, :, l] = B[:, :, l + 1] + (self.alpha / (self.L * self.M * self.D)) * (self.U[:, :, l] @ Q[:,:, l])
        return B, Q

    def compute_gradients(self, P, H, B, Q):
        gradU = np.zeros_like(self.U)
        gradV = np.zeros_like(self.V)

        for l in range(self.L):
            gradU[:, :, l] = (1 / self.D) * (H[:, :, l] @ Q[:, :, l].T)
            gradV[:, :, l] = B[:, :, l + 1] @ self.sigma(P[:, :, l]).T

        return gradU, gradV

    def step(self, X, Y, eta, eta_u=1.0, eta_v=1.0):
        # 1. Forward Pass
        H, P = self.forward(X)
        D, N = X.shape
        out = H[:, :, self.L]

        # 2. Backward Pass (Compute B)
        W = self.loss_der(out, Y) / (D)
        B, Q = self.backward(W, H, P)

        # 3. Compute Gradients
        gradU, gradV = self.compute_gradients(P, H, B, Q)

        # 4. Apply Update (using -= as U and V are instance variables)
        self.U -= eta_u * (eta / self.alpha) * (1/N) * gradU
        self.V -= eta_v * (eta / self.alpha) * (1/N) * gradV
        
        current_loss = self.loss(out, Y) / (N*D)
        return current_loss