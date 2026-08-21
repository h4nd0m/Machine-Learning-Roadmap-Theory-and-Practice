import numpy as np

def loss_function(w):
    # w — это массив numpy размера (2,)
    # L(w1, w2) = w1^2 + 3*w1*w2 + w2^4
    return w[0]**2 + 3*w[0]*w[1] + w[1]**4

def numerical_gradient(f, w, h=1e-5):
    """
    f: функция, принимающая вектор w и возвращающая число
    w: numpy массив текущих параметров (точки, в которой считаем градиент)
    """
    grad = np.zeros_like(w, dtype=float) # Сюда будем записывать частные производные
    w_plus = w.copy()
    w_minus = w.copy()
    for i in range(len(w)):
        w_plus[i] += h
        w_minus[i] -= h
        grad[i] = (f(w_plus) - f(w_minus)) / (2 * h)
        w_plus[i] -= h
        w_minus[i] += h
    return grad

# Проверка
current_w = np.array([2.0, 1.0])
grad_num = numerical_gradient(loss_function, current_w)

# Аналитический градиент для проверки (выведи сам на бумаге и сравни):
# dL/dw1 = 2*w1 + 3*w2  => в точке (2,1) равно 2*2 + 3*1 = 7
# dL/dw2 = 3*w1 + 4*w2^3 => в точке (2,1) равно 3*2 + 4*1 = 10
grad_analytical = np.array([7.0, 10.0])

print(f"Численный градиент: {grad_num}")
print(f"Аналитический градиент: {grad_analytical}")
print(f"Ошибка: {np.linalg.norm(grad_num - grad_analytical)}")