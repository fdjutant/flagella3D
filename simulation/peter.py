import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

sx = 1
sy = 2

zs = np.linspace(0, 3, 300)
def fn(x, z):
    return np.exp(-x**2 / 2 / sx**2) * np.exp(-z**2 / 2 / sy**2 / x**2)

rs = np.zeros(len(zs))
for ii in range(len(zs)):
    rs[ii], unc = quad(lambda x: fn(x, zs[ii]), -np.inf, np.inf)

zs = np.concatenate((-np.flip(zs[1:]), zs))
rs = np.concatenate((np.flip(rs[1:]), rs))

# normalize
rs /= np.trapz(rs, zs)

figh = plt.figure()
plt.title("PDF of product of uncorrelated gaussians with $\sigma_x$=%0.2f, $\sigma_y$=%0.2f" % (sx, sy))
plt.plot(zs, rs)
plt.xlabel("result")
plt.ylabel("PDF")