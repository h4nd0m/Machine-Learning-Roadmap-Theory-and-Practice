import numpy as np

a = np.array([1, 2, 3, 4])
b = np.array([10, 20, 30, 40])

print(a**3)
print(a*b) # только если размеры равны 
# поэлементное умножение не матричное!!!

# np.sin np.log np.exp np.sqrt
angles = np.array([0, np.pi/2, np.pi])
print(np.sin(angles))

matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

row_vector = np.array([10, 20, 30])

result = matrix + row_vector
print(result)

