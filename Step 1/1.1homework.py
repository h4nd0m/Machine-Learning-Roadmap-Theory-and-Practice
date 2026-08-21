class SimpleKNN:
    def __init__(self, x_train, y_train):
        self.x = x_train
        self.y = y_train

    def change_data(self, new_x, new_y):
        self.x = new_x
        self.y = new_y

    def predict_closest(self, target_x):
        closest_pair = min(zip(self.x, self.y), key = lambda pair: abs(pair[0] - target_x))
        return closest_pair[1]
                
                
