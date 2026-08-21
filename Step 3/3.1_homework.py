import numpy as np
import pandas as pd

def target_function(x):
    # f(x) = ln(x) * e^x
    return np.log(x) * np.exp(x)

def analytical_derivative(x):
    return (1 / x) * np.exp(x) + np.log(x) * np.exp(x)
    pass

def right_difference(x, h):
    return (target_function(x+h) - target_function(x))/h
    pass

def central_difference(x, h):
    return (target_function(x+h) - target_function(x-h))/(2*h)
    pass

# Эксперимент
point = 2.0
point_result = analytical_derivative(point)
print(point_result)

steps = [1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
result = pd.DataFrame({
    'Шаг(h)': [],
    'Ошибка правой разности': [],
    'Ошибка центральной разности': []
})

for step in steps:
    result.loc[len(result)] = [step, abs(point_result - right_difference(2.0, step)), abs(point_result - central_difference(2.0, step))]

print(result)

# ТВОЙ КОД: В цикле пройдись по steps, посчитай ошибки и выведи таблицу