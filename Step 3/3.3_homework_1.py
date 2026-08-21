import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Генерация синтетического датасета для проверки
np.random.seed(42)
X_data = np.random.randn(1000, 3) # 1000 объектов, 3 признака
# Истинная линейная зависимость: y = 2*x1 - 3*x2 + 0.5*x3 + 5 + шум
true_w = np.array([2.0, -3.0, 0.5])
true_b = 5.0
y_data = X_data.dot(true_w) + true_b + np.random.randn(1000) * 0.1

class MiniBatchGradientDescentRegressor:
    def __init__(self, learning_rate=0.01, batch_size=64, epochs=20):
        self.lr = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.w = None # Вектор весов
        self.b = None # Смещение (bias)
        self.loss_history = [] # Сюда сохраняй средний MSE на каждой эпохе
        
    def fit(self, X, y):
        """
        Обучение модели.
        X: матрица признаков размера (N, m)
        y: вектор ответов размера (N,)
        """
        num_objects, num_features = X.shape
        
        # Инициализируем веса нулями
        self.w = np.zeros(num_features)
        self.b = 0.0
        
        for epoch in range(self.epochs):
            # Шаг 1: Перемешивание (Shuffle) данных в начале каждой эпохи,
            # чтобы батчи были случайными.
            indices = np.arange(num_objects)
            np.random.shuffle(indices)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_losses = []
            
            # Шаг 2: Цикл по мини-батчам
            for i in range(0, num_objects, self.batch_size):
                # ТВОЙ КОД: Вырежи текущий батч из X_shuffled и y_shuffled
                X_batch = X_shuffled[i:i+self.batch_size, :]
                y_batch = y_shuffled[i:i+self.batch_size]
                
                # ТВОЙ КОД: Сделай предсказание модели для этого батча: 
                # y_pred = X_batch * w + b
                y_pred = X_batch @ self.w + self.b
                
                # Вычисляем MSE для текущего батча и сохраняем для истории
                batch_loss = np.mean((y_pred - y_batch) ** 2)
                epoch_losses.append(batch_loss)
                
                # ТВОЙ КОД: Вычисли векторный градиент лосса по весам (w) и по смещению (b)
                # Вспоминай формулу: grad_w = (2 / len(batch)) * X_batch.T * (y_pred - y_batch)
                # Подумай, чему равна производная по свободному коэффициенту b!
                grad_w = (2 / self.batch_size) * X_batch.T @ (y_pred - y_batch)
                grad_b = 2 * np.mean(y_pred - y_batch)
                
                # ТВОЙ КОД: Сделай шаг градиентного спуска (обнови self.w и self.b)
                self.w = self.w - self.lr * grad_w
                self.b = self.b - self.lr * grad_b
                
            # Сохраняем средний лосс за всю эпоху
            self.loss_history.append(np.mean(epoch_losses))
            
    def predict(self, X):
        """Возвращает предсказания модели для матрицы X"""
        return X @ self.w + self.b

# 2. Тестирование твоего оптимизатора
model = MiniBatchGradientDescentRegressor(learning_rate=0.005, batch_size=32, epochs=30)
model.fit(X_data, y_data)

# Вывод результатов
print("--- Итоги обучения ---")
print(f"Истинные веса w: {true_w} | Полученные: {model.w}")
print(f"Истинный bias b: {true_b} | Полученный: {model.b}")
print(f"Финальный MSE: {model.loss_history[-1]}")

# 3. Эксперимент (необязательно, но круто для аналитики):
# Построй график изменения model.loss_history от номера эпохи. 
# Он должен красиво падать вниз и выходить на плато.
plt.figure(figsize=(8, 5))
plt.plot(model.loss_history, color='blue', linewidth=2, marker='o')
plt.title('Падение MSE при обучении', fontsize=14)
plt.xlabel('Эпоха', fontsize=12)
plt.ylabel('Среднеквадратичная ошибка(лосс)', fontsize=12)
plt.grid(True)
plt.show()