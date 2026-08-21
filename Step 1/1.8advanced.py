import pandas as pd
import numpy as np

# .apply() когда не хватает .sum() .mean()
# лямбда функция

df = pd.DataFrame({
    'City': ['Москва', 'Казань', 'Москва', 'Сочи', 'Казань'],
    'Sales': [100, 150, 200, 50, 300]
})

sales_range = df.groupby('City')['Sales'].apply(lambda x: x.max() - x.min())
print(sales_range)

# .apply() под капотом выполняется циклом так что
# всегда лучше стараться обходиться векторными функциями

df_time  = pd.DataFrame({
    'Sales': [10, 20, 15, 30, 45, 25]
})

df_time['Rolling_mean'] = df_time['Sales'].rolling(window = 3).mean()
print('')
print(df_time)

df_time['Cumulative_Sales'] = df_time['Sales'].cumsum()
print('')
print(df_time['Cumulative_Sales'])