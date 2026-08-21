import numpy as np
import pandas as pd

class MovieRecommender:
    def __init__(self, movies_dict):
        self.movies = movies_dict
    
    def _cosine_similarity(self, vec_a, vec_b):
        l2_norm_a = self.l2_norm(vec_a)
        l2_norm_b = self.l2_norm(vec_b)
        if l2_norm_a == 0 or l2_norm_b == 0:
            return 0
        return np.sum(vec_a*vec_b)/(l2_norm_a * l2_norm_b)
        
    def l2_norm(self, vec):
        return np.sqrt(np.sum(vec**2))
    
    def recommend(self, user_preferences):
        sorted_movies = sorted(self.movies.items(), 
                               key = lambda x: 
                               self._cosine_similarity(x[1], user_preferences), 
                               reverse = True)
        return np.array([sorted_movies[0][0], sorted_movies[1][0]])
    
if __name__ == '__main__':
    # База данных фильмов: [Экшен, Комедия, Драма, Фантастика]
    movies_db = {
        'Матрица': np.array([9.0, 1.0, 4.0, 10.0]),
        'Дэдпул': np.array([8.0, 10.0, 2.0, 7.0]),
        'Интерстеллар': np.array([5.0, 0.0, 9.0, 10.0]),
        '1+1': np.array([1.0, 9.0, 10.0, 0.0])
    }
    
    recommender = MovieRecommender(movies_db)
    
    # Пользователь любит глубокую драму и научную фантастику, не любит комедии
    user_profile = np.array([2.0, 0.0, 8.0, 9.0])
    
    recommended_movies = recommender.recommend(user_profile)
    print("Топ-2 рекомендации для пользователя:", recommended_movies)
    # Ожидаемый топ-1: Интерстеллар (косинус будет самым высоким)