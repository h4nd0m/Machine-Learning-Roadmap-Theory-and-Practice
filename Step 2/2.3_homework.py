import numpy as np

def check_linear_dependence(a, b):
    cos_sim = np.sum(a*b)/(np.sqrt(np.sum(a**2))*np.sqrt(np.sum(b**2)))
    if np.abs(cos_sim) > 0.99999:
        return True
    return False

def get_coordinates_in_onb(x, basis):
    c1 = np.sum(x*basis[0])
    c2 = np.sum(x*basis[1])
    c3 = np.sum(x*basis[2])
    return [c1, c2, c3]

if __name__ == '__main__':
    # Случай 1: Линейно зависимые векторы (b = 2.5 * a)
    v1 = np.array([2.0, -1.0, 4.0])
    v2 = np.array([5.0, -2.5, 10.0])
    print("Зависимы ли v1 и v2?:", check_linear_dependence(v1, v2)) # Ожидается True
    
    # Случай 2: Линейно независимые векторы
    v3 = np.array([1.0, 0.0, 0.0])
    v4 = np.array([0.0, 1.0, 0.0])
    print("Зависимы ли v3 и v4?:", check_linear_dependence(v3, v4)) # Ожидается False