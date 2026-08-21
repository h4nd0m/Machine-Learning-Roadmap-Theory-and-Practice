import numpy as np

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X_new):
        x_dist = self.X_train[np.newaxis, :, :] - X_new[:, np.newaxis, :]
        x_dist **= 2
        result_matrix = np.sqrt(np.sum(x_dist, axis=-1))

        nearest_index = np.argsort(result_matrix, axis=1)[:, :self.k]

        neighbor_labels = self.y_train[nearest_index]

        mean_neighbor = np.mean(neighbor_labels, axis=1)
        mean_neighbor[mean_neighbor>=0.5] = 1
        mean_neighbor[mean_neighbor<0.5] = 0

        return mean_neighbor