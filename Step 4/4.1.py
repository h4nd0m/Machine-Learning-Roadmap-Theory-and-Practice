import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.DataFrame(None)

# df.info()
# df.describe(столбец)
# df.value_counts(столбец)

# 1.

# график распределения цен сглаженной линией плотности (kde)
sns.histplot(data=df, x='Price', bins=30, kde=True)
plt.title("Распределение стоимости автомобилей")
plt.show()

# ящик с усами
sns.boxplot(data=df, x='Engine_Volume')
plt.title(...)
plt.show()

# 2. двумерный анализ(2 признака)

# зависимость цены от мощности двигателя
# объекты в виде точек на плоскости альфа - прозрачность
sns.scatterplot(data=df, x='Horsepower', y='Price', alpha=0.6)

# столбчатые
# количество объектов кажой категории
sns.countplot(data=df, x='Body_Style')

# среднее по каждой категории по умолчанию (или медиана или мода)
sns.barplot(data=df, x='Body_Style', y='Price')

# 3. многомерный анализ

# тепловая карта (корреляции Пирсона)
corr_matrix = df.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
# annot=True выводит цифры прямо на карту.
# cmap='coolwarm' красит сильную прямую связь в красный, 
# а сильную обратную — в синий.

# матрица графиков рассеяния
# Строит сетку, где на пересечении каждых двух признаков находится scatterplot, 
# а по диагонали (где признак пересекается сам с собой) — гистограмма
sns.pairplot(data=df[['Price', 'Horsepower', 'Age', 'Target']], hue='Target')
