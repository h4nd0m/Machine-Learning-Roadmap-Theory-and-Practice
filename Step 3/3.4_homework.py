import numpy as np
import matplotlib.pyplot as plt

def rosenbrock(w):
    """
    w: numpy массив из двух элементов [x, y]
    """
    x, y = w[0], w[1]
    return (1 - x)**2 + 100 * (y - x**2)**2

def rosenbrock_gradient(w):
    """
    Возвращает аналитический градиент функции Розенброка в точке w
    """
    x, y = w[0], w[1]
    df_dx = -2*(1-x)-400*x * (y-x**2)
    df_dy = 200*(y-x**2)
    
    return np.array([df_dx, df_dy])

def adam_descent(start_w, lr=0.05, epochs=1000, beta1=0.9, beta2=0.999, epsilon=1e-8):
    w = np.array(start_w, dtype=float)
    
    # Инициализация моментов (векторы из нулей той же формы, что и w)
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    
    loss_history = []
    trajectory = [w.copy()]
    
    # Важно: счетчик шагов t должен начинаться с 1 для корректной работы bias correction
    for t in range(1, epochs + 1):
        # 1. Получаем текущий градиент
        grad = rosenbrock_gradient(w)
        
        # 2. ТВОЙ КОД: Обнови первый момент m (инерция)
        m = beta1 * m + (1 - beta1) * grad
        
        # 3. ТВОЙ КОД: Обнови второй момент v (RMSprop, квадрат градиента)
        # Подсказка: возвести вектор в квадрат в NumPy можно как grad**2
        v = beta2 * v + (1 - beta2) * grad**2
        
        # 4. ТВОЙ КОД: Сделай bias correction (скорректируй m и v)
        # Подсказка: возведение числа в степень t это beta1**t
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        # 5. ТВОЙ КОД: Сделай шаг обновления весов w
        # Подсказка: корень из массива в NumPy это np.sqrt(...)
        w = w - lr*m_hat/(np.sqrt(v_hat) + epsilon)
        
        loss_history.append(rosenbrock(w))
        trajectory.append(w.copy())
        
    return loss_history, np.array(trajectory)

# --- ЗАПУСК И ВИЗУАЛИЗАЦИЯ ---
start_point = [-1.5, 2.0]
# Обрати внимание: lr = 0.05, что в 50 раз больше, чем в прошлом задании!
losses_adam, path_adam = adam_descent(start_point, lr=0.05, epochs=1500)

print(f"Стартовая точка: {start_point}")
print(f"Финальная точка после спуска (Adam): {path_adam[-1]}")
print(f"Значение функции в ней: {losses_adam[-1]}")

# Построй такие же два графика, как в прошлом задании, передав в них losses_adam и path_adam.
# Если хочешь эстетики — нарисуй на графике траектории красную точку-цель в координатах (1, 1):
# ax2.scatter(1, 1, color='red', marker='x', s=100, label='Минимум')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(losses_adam, linewidth=2, color='red')
ax1.set_title('Потери Adam', fontsize=12)
ax1.set_xlabel('Эпоха', fontsize=12)
ax1.set_ylabel('Значение ошибки', fontsize=12)
ax1.grid(True)

ax2.plot(path_adam[:, 0], path_adam[:, 1], linewidth=2, color='blue')
ax2.set_title('Траектория Adam', fontsize=12)
ax2.set_xlabel('Ось X', fontsize=12)
ax2.set_ylabel('Ось Y', fontsize=12)
ax2.grid(True)

plt.show()