import numpy as np

def count_paths_of_length_k(A, k):
    return np.linalg.matrix_power(A, k)

def total_paths_up_to_k(A, start, end, k):
    res = 0
    for i in range(1, k+1):
        res += np.linalg.matrix_power(A, i)[start, end]
    return res
