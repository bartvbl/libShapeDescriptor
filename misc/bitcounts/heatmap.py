import matplotlib.pyplot as plt
import numpy as np
import sys

a = np.loadtxt(open(sys.argv[1], "rb"), delimiter=",", skiprows=0)

plt.imshow(a, cmap='hot', interpolation='nearest')
plt.show()