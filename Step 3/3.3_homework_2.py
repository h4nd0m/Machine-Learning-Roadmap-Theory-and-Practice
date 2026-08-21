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
    
    # ТВОЙ КОД: Посчитай df/dx и df/dy по формулам выше
    df_dx = -2*(1-x)-400*x * (y-x**2)
    df_dy = 200*(y-x**2)
    
    return np.array([df_dx, df_dy])

def gradient_descent(start_w, lr=0.001, epochs=5000):
    w = np.array(start_w, dtype=float)
    
    # Сюда мы будем сохранять историю для графиков
    loss_history = []
    trajectory = [w.copy()] # Список векторов [x, y] на каждом шаге
    
    for epoch in range(epochs):
        # 1. ТВОЙ КОД: Посчитай градиент в текущей точке w
        grad = rosenbrock_gradient(w)
        
        # 2. ТВОЙ КОД: Сделай шаг градиентного спуска (обнови w)
        w = w - lr * grad
        
        # Сохраняем историю
        loss_history.append(rosenbrock(w))
        trajectory.append(w.copy())
        
    return loss_history, np.array(trajectory)

# --- ЗАПУСК ОПТИМИЗАЦИИ ---
start_point = [-1.5, 2.0]
# Важно: для функции Розенброка нужен маленький learning rate (например, 0.0005 или 0.001), 
# иначе она мгновенно улетит в бесконечность!
losses, path = gradient_descent(start_point, lr=0.001, epochs=6000)

print(f"Стартовая точка: {start_point}")
print(f"Финальная точка после спуска: {path[-1]}")
print(f"Значение функции в ней: {losses[-1]}")


# --- ТВОЙ КОД: ВИЗУАЛИЗАЦИЯ (МАТПЛОТЛИБ) ---
# Давай сделаем красивое окно, разделенное на два графика (1 строка, 2 колонки)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# ГРАФИК 1: Падение лосса (используй объект ax1 вместо plt)
# ТВОЙ КОД: Нарисуй линию изменения losses. 
# Сделай ее красной ('red'), толщиной 2. Маркеры не ставь, точек слишком много (6000 штук!).
ax1.plot(losses, color='red', linewidth=2) 
ax1.set_title('Падение значения функции (Loss)', fontsize=12)
ax1.set_xlabel('Итерация')
ax1.set_ylabel('Значение функции')
ax1.grid(True)

# ГРАФИК 2: Траектория спуска (используй объект ax2 вместо plt)
# Мы хотим увидеть, как точка ползла по координатам.
# path[:, 0] — это все иксы нашей траектории, path[:, 1] — все игреки.
# ТВОЙ КОД: Отрисуй линию движения: ax2.plot(path[:, 0], path[:, 1], ...)
# Сделай ее зеленой ('green'), добавь подписи осей x и y через ax2.set_xlabel() и ax2.set_ylabel()
ax2.plot(path[:, 0], path[:, 1], color='green', linewidth=2)
ax2.set_title('Траектория движения в пространстве XY', fontsize=12)
ax2.set_xlabel('Ось X', fontsize=12)
ax2.set_ylabel('Ось Y', fontsize=12)
ax2.grid(True)

# Показываем шедевр на экран
plt.show()