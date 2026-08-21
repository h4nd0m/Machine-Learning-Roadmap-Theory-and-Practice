import numpy as np

def get_matrix_properties(a):
    return (np.linalg.det(a), np.linalg.matrix_rank(a))

def is_matrix_invertible(a):
    det = get_matrix_properties(a)[0]
    return not np.isclose(det, 0)