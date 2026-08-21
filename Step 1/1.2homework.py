class BaseMetric:
    def __init__(self):
        self.__metric_type = "Standart"
    def get_type(self):
        return self.__metric_type
    def score(self, y_true, y_pred):
        return 0.0

class MAEMetric(BaseMetric):
    def __init__(self, round_digits=2): #=2 значение по умолчанию
        super().__init__() #наследование всех аргументов из родительского класса
        self.digits = round_digits
    def score(self, y_true, y_pred):
        return sum(abs(y_true[i]-y_pred[i]) for i in range(len(y_pred)))/len(y_pred)

class MSEMetric(BaseMetric):
    def score(self, y_true, y_pred):
        return sum((y_true[i]-y_pred[i])**2 for i in range(len(y_pred)))/len(y_pred)