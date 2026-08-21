import numpy as np
import pandas as pd

df = pd.DataFrame({
        'Product': ['Монитор', 'Клавиатура', 'Мышь', 'Ноутбук', 'Наушники'],
        'Amount': [10, 50, 65, 5, 32],
        'Price': [25000.0, 3500.0, 1800.0, 95000.0, 5500.0]
    })

df_named = df.set_index('Product')

mouse_price = df_named.loc['Мышь', 'Price'] 
# вернёт 1800

sub_matrix = df_named.loc['Клавиатура':'Ноутбук', 'Amount':'Price']
# можно использовать срезы по именам
# print(sub_matrix)

first_val = df.iloc[0, 0]

sub_df = df.iloc[1:3, 0:2]
print(sub_df)

# оптимизация памяти .astype()
# целым изначально даёт int64
# можно понизить до int32/16/8
# float64 -> float 32
# object(строки) если много раз поторяются 
# надо перевести в category
