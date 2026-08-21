class FunctionLayer:
    def __init__(self, name):
        self.name = name
    def __str__(self):
        return f"[Слой: {self.name}]"
    def __call__(self, x):
        return x

class Pipeline:
    def __init__(self):
        self.layers = []
    def add_layer(self, layer):
        self.layers.append(layer)
    def __str__(self):
        res=" -> ".join(str(x) for x in self.layers)
        return "Pipeline:"+res
    def __call__(self, x):
        for i in range(len(self.layers)):
            x=self.layers[i](x)
        return x

class SquareLayer(FunctionLayer):
    def __call__(self, x): return x ** 2

class IncrementLayer(FunctionLayer):
    def __call__(self, x): return x + 10

if __name__ == "__main__":
    pipe = Pipeline()
    pipe.add_layer(SquareLayer("Возведение в квадрат"))
    pipe.add_layer(IncrementLayer("Прибавление 10"))
    
    print(pipe)