import numpy as np

def einsum_dot(a, b):
    return np.einsum('i,i->', a, b)

def einsum_elementwise(a, b):
    return np.einsum('ij,ij->ij', a, b)

def einsum_bmm(x, y):
    return np.einsum('bnm,bmk->bnk', x, y)