import numpy as np

def pagerank(A, d=0.85, tol=1e-6, max_iter=100):
    P = A.astype(float)
    N = P.shape[0]

    row_sums = np.sum(P, axis=1)
    
    dangling_mask = (row_sums == 0)
    normal_mask = ~dangling_mask

    P[normal_mask] /= row_sums[normal_mask][:, None]
    P[dangling_mask] = 1.0/N

    G = d*P + (1.0 - d) / N * np.ones((N, N))

    pi = np.ones(N)/N

    for _ in range(max_iter):
        pi_new = pi @ G
        if np.linalg.norm(pi_new - pi) <tol:
            break
        pi = pi_new

    return pi_new

