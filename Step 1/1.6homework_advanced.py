import numpy as np

class LinearRegressor:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def predict(self, X_matrix):
        return (X_matrix @ self.weights) + self.bias

    def fit(self, X_matrix, Y_vector):
        xtx = X_matrix.T @ X_matrix
        xtx_inv = np.linang.inv(xtx)
        self.weights = xtx_inv @ X_matrix.T @ Y_vector
