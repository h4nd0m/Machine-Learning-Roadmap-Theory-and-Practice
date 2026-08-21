import numpy as np

class VectorMetrics:
    def dot_product(self, a, b):
        return np.sum(a*b)
    
    def l1_norm(self, x):
        return np.sum(np.abs(x))
    
    def l2_norm(self, x):
        return np.sqrt(np.sum(x**2))

    def minkovski_distance(self, a, b, p):
        return (np.sum((np.abs(a-b))**p))**(1/p)
    
    def normalize(self, x):
        norm = self.l2_norm(x)
        if norm == 0:
            print("ZeroDivisionError")
            return x
        return x/(norm)
    
    def mean_absolute_error(self, y_true, y_pred):
        return np.abs(y_true - y_pred).mean()

    pass


if __name__ == '__main__':
    metrics = VectorMetrics()
    
    vec_a = np.array([3.0, -4.0, 0.0])
    vec_b = np.array([1.0, 2.0, 3.0])
    
    # 1. Скалярное произведение: 3*1 + (-4)*2 + 0*3 = 3 - 8 + 0 = -5.0
    print("Dot Product:", metrics.dot_product(vec_a, vec_b))
    
    # 2. L1 норма vec_a: |3| + |-4| + |0| = 7.0
    print("L1 Norm:", metrics.l1_norm(vec_a))
    
    # 3. L2 норма vec_a: sqrt(3^2 + (-4)^2 + 0^2) = sqrt(9 + 16) = 5.0
    print("L2 Norm:", metrics.l2_norm(vec_a))