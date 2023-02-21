import numpy as np

R = 1
L = 30.9
wavelen = 10.2
a = 0.04

# sbt, then stokeslet, then sbt
bda_mat = np.array([[11.9275, 87.7748, 41.5752],
                    [12.011, 90.7083, 42.6751],
                    ])

eff = bda_mat[:, 0] ** 2 / (4 * bda_mat[:, 1] * bda_mat[:, 2])

bda_star = np.array(bda_mat, copy=True)
bda_star[:, 0] *= R**2 / (L * R)
bda_star[:, 1] *= R**3 / (L * R ** 2)
bda_star[:, 2] *= R / (L)