import numpy as np

class LinearRegressor:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def predict(self, X_matrix):
        return (X_matrix @ self.weights) + self.bias
