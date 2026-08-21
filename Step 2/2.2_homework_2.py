import numpy as np

def matrix_recommend(movies_matrix, movie_names, user_preferences):
    u_l2_norm = np.sqrt(np.sum(user_preferences**2))
    u_norm = user_preferences/u_l2_norm

    matrix2 = movies_matrix**2
    movies_matrix_norm = np.sqrt(np.sum(matrix2, axis=1, keepdims=True))
    m_norm = movies_matrix/movies_matrix_norm

    cos_matrix = m_norm @ u_norm
    sorted_cos_matrix = np.argsort(cos_matrix)[::-1]
    return movie_names[sorted_cos_matrix[:2]]

def get_projection(a, b):
    l2_b = np.sqrt(np.sum(b**2))
    return b*(np.sum(a*b)/l2_b)


