import numpy as np

def analyltical_ols(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

def pinv_ols(X, y):
    return np.linalg.pinv(X) @ y

def ridge_ols(X, y, alpha=0.1):
    return np.linalg.inv((X.T @ X + np.eye(X.shape[1])*alpha)) @ X.T @ y

# New task

def remove_collinear_features(X, threshold = 0.999):
    indices_of_column = [0]
    norms = np.linalg.norm(X, axis = 0)

    for i in range(1, X.shape[1]):
        column = X[:, i]
        norm_i = norms[i]

        if norm_i == 0:
            continue

        approved_cols = X[:, indices_of_column]
        approved_norms = norms[indices_of_column]

        dot_products = column @ approved_cols
        cos_sim = np.abs(dot_products / (norm_i * approved_norms))

        if np.all(cos_sim <= threshold):
            indices_of_column.append(i)
  
    return X[:, indices_of_column]

