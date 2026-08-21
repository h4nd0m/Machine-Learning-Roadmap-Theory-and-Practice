import numpy as np

class BatchNormalizer:
    def __init__(self, batch_tensor):
        self.batch = batch_tensor
    
    def normalize_channels(self, mean_vector):
        return self.batch - mean_vector[np.newaxis, np.newaxis, np.newaxis, :]
    