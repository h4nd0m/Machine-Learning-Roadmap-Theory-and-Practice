import numpy as np

sales = np.array([
    [100, 500],
    [200, 600],
    [300, 700]
])

day_means = np.mean(sales, axis = 1)
print(day_means)

column_means = day_means[:, np.newaxis]
print(column_means.shape)
print(column_means)

centered_by_days = sales - column_means

# .reshape .transpose

# .reshape галовное чтобы 
# общее количество элементов осталось тем же
print("")
arr = np.arange(6)

matrix = arr.reshape(2, 3) # можно поставить -1 в одно из полей
# и numpy сам посчитает что там должно стоять 
print(matrix)

# .T or .transpose

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6]
])


