import numpy as np

a = np.array([[1,2,3]])
b = np.array([[1,2,3,4]])

print(a.shape)
print(b.shape)
print(np.concatenate([a,b], axis=0))