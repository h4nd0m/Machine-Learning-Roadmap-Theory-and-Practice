import numpy as np
import pandas as pd

class AdvancedAnalytics:
    def __init__(self, df_transactions):
        self.df = df_transactions
    
    def add_user_rolling_mean(self, window_size=2):
        self.df['user_rolling_mean'] = self.df.groupby('user_id')['amount'].transform(lambda x: x.rolling(window = window_size, min_periods=1).mean())

    def get_user_peak_diff(self):
        return self.df.groupby('user_id')['amount'].apply(lambda x: x.max()-x.mean())
