import numpy as np
class DataScaler:
    def __init__(self, data_matrix):
        self.data = data_matrix

    def min_max_scaler(self):
        data_min = np.min(self.data, axis = 0)
        data_max = np.max(self.data, axis = 0)

        data_res1 = self.data - data_min
        data_res2 = self.data - data_max

        return data_res1/data_res2
    
    def add_bias(self, bias_vector):
        return self.data + bias_vector
    

    