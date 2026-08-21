class Model:
    pass

my_model = Model()
print(type(my_model))

class LinearModel:
    def __init__(self, w_init, b_init):
        self.weight = w_init
        self.bias = b_init
    def predict(self, x):
        return self.weight*x+self.bias

model_1 = LinearModel(w_init = 2.5, b_init = 0.1)

result = model_1.predict(2.0)

print(result)

