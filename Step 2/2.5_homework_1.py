import numpy as np

def transform_data(data, transform_matrix):
    return data @ transform_matrix.T

def rotate_data(data, angle_degrees):
    rad = np.radians(angle_degrees)
    r_matrix = np.array([
        [np.cos(rad), -np.sin(rad)],
        [np.sin(rad), np.cos(rad)]
    ])
    return data @ r_matrix.T

def f(a, b):
    c1 = a @ b
    c2 = b @ a
    vec = np.array([1.0, 1.0])
    return (vec @ c1.T), (vec @ c2.T)
