import matplotlib.pyplot as plt
import numpy as np
from gpy.kernels import PeriodicKernel

# generate data to use with the kernel
x = np.linspace(start=0, stop=10, num=100)

# instantiate Periodic Kernel
# try running this with different hyperparameter values to see how it changes
# the plot
kernel = PeriodicKernel(sigma=1.0, length_scale=1.0, period=2.0 * np.pi)

# use the kernel's "compute" method to generate the covariance matrix over x
sim_matrix = kernel.compute(x, x)

# plot heat map of the similarity matrix
plt.imshow(sim_matrix, extent=[x.min(), x.max(), x.min(), x.max()])
plt.xlabel("x")
plt.ylabel("x")
plt.title("Periodic Kernel Similarity Matrix")
plt.colorbar()
plt.show()
