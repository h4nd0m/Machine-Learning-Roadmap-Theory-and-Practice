# __str__

class Model:
    def __init__(self, name):
        self.name = name
    
    def __str__(self):
        return f"Модель: {self.name}"

model = Model("XGBoost")
print(model)

# __call__

class Multiplier:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return x * self.factor
    
tripler = Multiplier(factor=3)

print(tripler(5)) #15

# множественное наследование

class Worker:
    def work(self):
        print("Я работаю")

class Student:
    def study(self):
        print("Я учусь")

class Intern(Worker, Student):
    pass

intern = Intern()
intern.work()
intern.study()

class A:
    def process(self): print("Process в классе A")

class B(A):
    def process(self): print("Process в классе B")

class C(A):
    def process(self): print("Process в классе C")

# Наследуемся от B и C. 
# Порядок в скобках имеет решающее значение!
class D(B, C):
    pass

obj = D()
obj.process()  # Выведет: Process в классе B (так как B указан первым)

# Посмотрим точный порядок поиска:
print(D.mro()) 
# Выведет список: [D, B, C, A, object]

# Python сначала ищет метод в самом D, 
# если не нашел — идет в B, если не нашел — в C, 
# и только потом поднимается к общему предку A.