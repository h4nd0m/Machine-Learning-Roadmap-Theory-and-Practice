import numpy as np

x = np.array([
    [1.0, 2.0], 
    [3.0, 4.0],
    [5.0, 6.0]
])

y = np.array([5.0, 11.0, 17.0])

xtx = x.T @ x
xtx_inv = np.linalg.inv(xtx)
w_optimal = xtx_inv @ x.T @ y

print(w_optimal)

batch_A = np.ones((2, 3, 4))
batch_B = np.ones((2, 4, 5))

result = batch_A @ batch_B
print(result.shape)