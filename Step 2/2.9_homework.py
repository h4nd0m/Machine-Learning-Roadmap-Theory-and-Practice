import numpy as np

def verify_spectrum_orthogonality(X):
    matrix_cov_x = np.cov(X.T)
    eig_vectors = np.linalg.eig(matrix_cov_x)[1]
    return np.isclose(eig_vectors[:, 0] @ eig_vectors[:, 1], 0)

def compress_features(X, n_componets=1):
    X_centred = X - np.mean(X, axis=0)
    matrix_cov_x = np.cov(X_centred.T)
    eig_values_cov_x, eig_vectors_cov_x = np.linalg.eig(matrix_cov_x)
    sorted_eig_values = np.argsort(eig_values_cov_x)[::-1]
    matrix_w = eig_vectors_cov_x[:, sorted_eig_values[:n_componets]]
    X_compressed = X_centred @ matrix_w
    return X_compressed


