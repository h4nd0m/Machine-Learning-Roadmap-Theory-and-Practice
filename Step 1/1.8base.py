import numpy as np
import pandas as pd

df = pd.DataFrame({
    'City': ['Москва', 'Казань', 'Москва', 'Сочи'],
    'Sales': [100, 150, 200, 50]
})

moscow_df = df[df['City'] == 'Москва']
print(moscow_df)
print('')
# работают условные операторы & и |
heavy_sales = df[(df['City'] == 'Москва') & (df['Sales'] > 120)]
print(heavy_sales)

# df.groupby('колонка_для_группировки')
# ['колонка_для_подсчета'].функция()
tcs = df.groupby('City')['Sales'].sum()
print('')
print(tcs)
tcs1 = df.groupby('City')['Sales'].agg(['sum', 'mean'])
print('')
print(tcs1)

print('')
# pd.merge()

# Таблица с заказами
orders = pd.DataFrame({
    'order_id': [1, 2, 3],
    'user_id': [101, 102, 101],
    'amount': [2500, 3400, 1200]
})

# Таблица с пользователями
users = pd.DataFrame({
    'user_id': [101, 102, 103],
    'name': ['Алексей', 'Мария', 'Иван']
})

merged_df = pd.merge(orders, users, on='user_id', how='inner')
print(merged_df)
