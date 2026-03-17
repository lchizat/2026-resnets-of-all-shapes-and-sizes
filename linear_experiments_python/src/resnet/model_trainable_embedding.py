import numpy as np

from src.resnet.model import FiniteResNet
from .activations import tanh, tanh_der
from .losses import quadratic_loss, quadratic_loss_der

class FiniteResNetWithTrainableEmbedding(FiniteResNet):
    def __init__(self, D, M, L, d_in, d_out, alpha=1.0, activation=tanh, activation_der=tanh_der, loss=quadratic_loss, loss_der=quadratic_loss_der):
        super().__init__(D, M, L, alpha, activation, activation_der, loss, loss_der)
        self.W_e = np.random.randn(D, d_in) # assuming fixed variance 1 
        self.W_u = np.random.randn(D, d_out) # assuming fixed variance 1

    def forward(self, X):
        d_in, N = X.shape
        E = self.W_e @ X
        
        H, P = super().forward(E)
        
        out = (1/self.D)*(self.W_u.T @ H[:, :, self.L])

        return out, H, P

    def compute_embedding_gradients(self, H, B, X, nabla_f):

        gradW_u = (H[:, :, self.L] @ nabla_f.T) # (D,N)@(N,d_out)
        gradW_e = B[:, :, 0] @ X.T # (D,N)@(N,d_in)

        return gradW_e, gradW_u

    def step(self, X, Y, eta, eta_u=1.0, eta_v=1.0, eta_we=1.0, eta_wu=1.0):
        # 1. Forward Pass
        out, H, P = self.forward(X)
        D, N = H[:, :, self.L].shape

        # 2. Backward Pass (Compute B)
        # Init condition: b(final) = gradient of loss w.r.t. final output
        nabla_f = self.loss_der(out, Y)
        W = (1/D) * self.W_u @ nabla_f
        B, Q = self.backward(W, H, P)

        # 3. Compute gradients
        gradU, gradV = self.compute_gradients(P, H, B, Q)
        gradW_e, gradW_u = self.compute_embedding_gradients(H, B, X, nabla_f)

        # 4. Apply Update (using -= as U and V are instance variables)
        self.U -= eta_u * (eta / self.alpha) * (1/N) * gradU
        self.V -= eta_v * (eta / self.alpha) * (1/N) * gradV
        self.W_e -= eta_we * (eta / self.alpha) * (1/N) * gradW_e
        self.W_u -= eta_wu * (eta / self.alpha) * (1/N) * gradW_u
        
        current_loss = (1/N)*self.loss(out, Y) # loss gives the sum over n
        return current_loss