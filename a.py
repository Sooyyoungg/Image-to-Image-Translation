import numpy as np
a = [[1,2,3,4], [5,6,7,8]]
b = [[11,12,13,14],[15,16,17,18]]
c = np.concatenate([a, b])
print(c)
print(c[:3, :])
print(c[3:, :])