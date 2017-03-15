import numpy as np
import matplotlib.pyplot as plt
import george
from george.kernels import ExpSquaredKernel

# Generate some fake noisy data.
x = 10 * np.sort(np.random.rand(50))
yerr = 0.2 * np.ones_like(x)
y = np.sin(x) + yerr * np.random.randn(len(x))

plt.errorbar(x,y,yerr=yerr,linestyle='none',marker='o')
plt.show()

# Set up the Gaussian process.
kernel = ExpSquaredKernel(1.0)
gp = george.GP(kernel)

print(type(x))
# Pre-compute the factorization of the matrix.
gp.compute(x)

# Compute the log likelihood.
print(gp.lnlikelihood(y))

t = np.linspace(0, 10, 500)
mu, cov = gp.predict(y, t)
std = np.sqrt(np.diag(cov))