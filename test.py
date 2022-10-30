import time
import numpy as np

start = time.time()

a = np.array([0])
for i in range(100000000):
    a[0] = i
end = time.time()
print("Numpy time is: ", (end-start))

start = time.time()
a = [0]
for i in range(100000000):
    a[0] = i
end = time.time()
print("List time is: ", (end-start))