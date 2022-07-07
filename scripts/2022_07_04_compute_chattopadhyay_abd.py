"""
Compute A, B, D from https://www.pnas.org/doi/full/10.1073/pnas.0602043103
"""
import numpy as np

# values given in results section
eta = 1e-3 # Pa.sec
theta = 41 * np.pi / 180
wavelen = 1.5e-6 # pitch in um
R = np.tan(theta) * wavelen / (2*np.pi)
L = 6.5e-6 # length in um
a = 0.02e-6 # filament radius

A = 1.48e-8 # N * s *m / m
A_unc = 0.04e-8
B = 7.9e-16 # N * s
B_unc = 0.2e-15
D = 7.0e-22 # N * s * m =
D_unc = 0.1e-22

a_star = A / L / eta
a_star_unc = A_unc / L / eta
b_star = B / (L * R) / eta
b_star_unc = B_unc / (L * R) / eta
d_star = D / (L * R**2) / eta
d_star_unc = D_unc / (L * R**2) / eta

print("averaged experimental values")
print(f"B*={b_star:.3f}({b_star_unc * 1e3:.0f})")
print(f"D*={d_star:.3f}({d_star_unc * 1e3:.0f})")
print(f"A*={a_star:.3f}({a_star_unc * 1e3:.0f})")