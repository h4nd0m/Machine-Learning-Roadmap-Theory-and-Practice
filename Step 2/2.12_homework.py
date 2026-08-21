import numpy as np

def svd_compress(A, k):
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    U_clipped = U[:, :k]
    s_clipped = np.diag(s[:k])
    Vt_clipped = Vt[:k, :]
    return U_clipped @ s_clipped @ Vt_clipped
    