"""
Estimate diffusion coefficients for finite cylinder of length L and diameter d based on
https://doi.org/10.1063/1.1730995
and
https://doi.org/10.1063/1.441071
"""
import numpy as np

def gamma_perp(sigma):
    return -0.193 + 0.15 / sigma + 8.1 / sigma**2 - 18 / sigma**3 + 9 / sigma**4

def gamma_par(sigma):
    return 0.807 + 0.15 / sigma + 13.5 / sigma**2 - 37 / sigma**3 + 22 / sigma**4

def d_parallel_to_perp(L, d):
    sigma = np.log(2 * L / d)
    return 2 * (sigma - gamma_par(sigma)) / (sigma - gamma_perp(sigma))


l_flagella = 7.7
d_flagella = 0.02
theta_flagella = 32 * np.pi / 180
arclen_flagella = l_flagella / np.cos(theta_flagella)

d_parallel_to_perp(arclen_flagella, d_flagella)
d_parallel_to_perp(l_flagella, d_flagella)