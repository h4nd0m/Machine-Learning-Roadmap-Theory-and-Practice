# Fraud Detection System
import numpy as np
import pandas as pd

class FraudDetector:
    def __init__(self, df_transactions, df_users):
        self.df_transactions = df_transactions
        self.df_users = df_users

    def prepare_data(self):
        sorted_df_transactions = self.df_transactions[self.df_transactions['is_declined'] == 0]
        self.df = pd.merge(sorted_df_transactions, self.df_users, on='user_id', how='inner')

    def analyze_and_detect(self, threshold_multiplier=2.0):
        self.df['user_rolling_mean'] = self.df.groupby(['user_id'])['amount'].transform(
            lambda x: x.rolling(window=2, min_periods=1).mean()
        )

        global_mean = self.df['amount'].mean()

        self.df['is_fraud'] = ((self.df['amount'] > self.df['user_rolling_mean']*threshold_multiplier) &
                               (self.df['amount'] > global_mean)).astype(int)
        
        return self.df
    
if __name__ == '__main__':
    # Сырые транзакции
    tx_data = pd.DataFrame({
        'user_id': [1, 1, 1, 2, 2, 2],
        'amount': [1000, 1500, 90000, 5000, 4000, 3000], 
        'is_declined': [0, 0, 0, 1, 0, 0] # У юзера 2 первая транзакция отклонена банком
    })

    # Профили пользователей
    user_data = pd.DataFrame({
        'user_id': [1, 2, 3],
        'age': [25, 42, 19],
        'risk_group': ['low', 'high', 'low']
    })

    # Инициализируем систему безопасности
    detector = FraudDetector(tx_data, user_data)

    # Шаг 1: Очистка и слияние
    detector.prepare_data()

    # Шаг 2: Анализ и поиск мошенников
    final_df = detector.analyze_and_detect(threshold_multiplier=2.5)

    print("--- Результаты работы системы Fraud Detection ---")
    print(final_df[['user_id', 'amount', 'user_rolling_mean', 'is_fraud']])


