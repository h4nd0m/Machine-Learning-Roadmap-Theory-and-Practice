import numpy as np

class DistanceCalculator:
    def __init__(self, X):
        self.x = X
    def compute_distance_matrix(self):
        x3 = self.x[:, np.newaxis, :]-self.x[np.newaxis, :, :]
        x3**=2

        result_matrix = np.sqrt(np.sum(x3, axis = 2), axis = 1)
        return result_matrix
 


    