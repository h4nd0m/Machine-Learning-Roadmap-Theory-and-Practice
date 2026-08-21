import numpy as np

vec = np.array([1, 2, 3, 4])
print(vec)
print(vec.dtype)

matrix = np.array([[1, 2, 3],[4, 5, 6]])
print(matrix.shape)

zeros_m = np.zeros((3, 3))
ones_m = np.ones((2, 5))

sequence = np.arange(start=0, stop=10, step=2) #range

arr = np.array([
    [10, 11, 12, 13],
    [20, 21, 22, 23],
    [30, 31, 32, 33],
    [40, 41, 42, 43]
])

print(arr[1, 2]) # первая строка второй столбец
print(arr[1, :]) # первая строка все элементы
print(arr[:, 1]) # весб первый столбец

# вырезать подматрицу
# arr[0:2, 1:3] строки 0 1, столбцы 1 2
 
print(arr.shape)
print(arr.ndim)
print(arr.size)
print(arr.dtype)

matrix_mut = np.zeros((3, 3))
matrix_mut[0, :] = 5
matrix_mut[1, :] = [1, 2, 3]
matrix_mut[1:3, 1:3]

print(matrix_mut[[0, 2], [1, 2]][0])

matrix = np.array([
    [10, 20, 30, 40],
    [50, 60, 70, 80],
    [90, 100, 110, 120]
])

sub_matrix = matrix[np.ix_([0, 2], [1, 3])]
print(sub_matrix)

data = np.array([1, 2, 3, 4, 5, 6])
mask = data > 3
print(mask)
print(data[mask]) 

matrix = np.array([[1, -2], 
                   [2, -3]])
matrix[matrix < 0] = 0
print(matrix)


