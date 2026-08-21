import numpy as np

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _compute_distances(self, X_test):
        X = X_test[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]
        X**=2
        X = np.sqrt(np.sum(X, axis=-1))
        return X
    
    def predict(self, X_test):
        X_indices = np.argsort(self._compute_distances(X_test), axis=-1)[:, :self.k]
        y_mean_vec = np.mean((self.y_train[X_indices]), axis=-1)
        return (y_mean_vec).astype(int)
    
if __name__ == '__main__':
    # Тренировочные данные (4 объекта, 2 признака)
    # Представь, что это [Возраст, Доход] (условно нормированные)
    X_train = np.array([
        [1.0, 1.1],
        [1.0, 1.0],
        [0.0, 0.0],
        [0.0, 0.1]
    ])
    y_train = np.array([1, 1, 0, 0]) # Классы объектов
    
    # Новые (тестовые) объекты, для которых нужно предсказать класс
    X_test = np.array([
        [0.9, 0.9], # Близок к группе со значением 1
        [0.1, 0.0]  # Близок к группе со значением 0
    ])
    
    knn = KNNClassifier(k=3)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    
    print("Предсказания модели:", predictions) # Ожидается: [1, 0]