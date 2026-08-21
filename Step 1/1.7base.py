import pandas as pd
import numpy as np

temperatures = pd.Series([22.5, 24.0, 19.8], index = ['Пн', 'Вс', 'Ср'])
print(temperatures["Вс"])

data = {
    'Name': ['Anna', 'Igor', 'Oleg'],
    'Math_Score': [92, 78, 85],
    'City': ['Moscow', "St-P", 'Kazan']
}

df = pd.DataFrame(data)
print(df)

# .head(n) первые n строк

# .tail(n) последние n строк

# .info() — важнейший метод. 

# Показывает общую информацию: 
# сколько всего строк, какие есть колонки, 
# типы данных в них 
# (int64, float64, object для строк) 
# и есть ли пропуски (Non-Null).

# .describe() - выдаёт стату по всем колонкам

