#инкапсулация

class NeuralNetwork:
    def __init__(self):
        self.layers = 3
        self._framework = "PT"
        self.__weights = [0.1]

nn = NeuralNetwork()
print(nn.layers)
print(nn._framework)
#print(nn.__weights)

#наследование

class BaseModel:
    def __init__(self, model_name):
        self.name = model_name

    def print_info(self):
        print(f"Это ML под названием: {self.name}")

class RegressionModel(BaseModel):
    def __init__(self, model_name, loss_function):
        super().__init__(model_name)
        self.loss = loss_function
    
    def train(self):
        print(f"Обучаем модель {self.name} с функцией потерь {self.loss}")

model = RegressionModel(model_name = "MyLinearReg", loss_function = "MSE")

model.print_info()

model.train()

#полиморфизм

class DecisionTree(BaseModel):
    def print_info(self):
        print(f"Это Дерево Решений: {self.name}")
    
class NeuralNet(BaseModel):
    def print_info(self):
        print(f"Это Нейросеть: {self.name}")

def run_model_presentation(any_model):
    any_model.print_info()

tree = DecisionTree("Tree_v1")
net = NeuralNet("ResNet50")

run_model_presentation(tree)
run_model_presentation(net)