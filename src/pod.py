import numpy as np
from scipy import linalg
import torch as tc

class POD:

    def __init__(self, X, n_dims):
        self.X = X
        self.n_dims = n_dims

    def transform(self):
        #Center data
        self.mean = np.mean(self.X, axis=0).reshape(-1,1)
        self.Xnorm = self.X - self.mean.T

        self.U, self.S, self.V = linalg.svd(self.Xnorm, full_matrices=True)
        self.U_reduced = self.U[:,:self.n_dims]
        self.S_reduced = self.S[:self.n_dims]
        self.V_reduced = self.V[:self.n_dims,:]

        self.a = self.Xnorm @ self.V_reduced.T
        return self.a

    def inverse(self, a_all):
        m, n = np.shape(self.X)
        W = np.array([]).reshape(0, n)
        for a in a_all:
            W_new = (self.V_reduced.T @ a.reshape(-1,1) + self.mean).T
            W = np.concatenate((W, W_new))
        return W

    def neuralPOD(self):
        """Returns the reduced order POD basis function and the matrix of coefficients"""
        self.Q_reduced = self.U_reduced @ np.diag(self.S_reduced)
        return self.V_reduced, self.Q_reduced

    def neuralInverse(self, network, params):
        W = np.zeros((np.size(params, 0), np.size(self.X, 1)))
        for i in range(0, np.size(params, 0)):
            inputs = tc.from_numpy(params[i, :].reshape(-1))
            coeffs = network(inputs.float())
            coeffs = coeffs.cpu().detach().numpy()
            W[i, :] = coeffs @ self.V_reduced + self.mean.T
        return W