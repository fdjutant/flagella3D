"""
Extracted data from https://www.pnas.org/doi/10.1073/pnas.94.21.11307
"""

import numpy as np

# #######################################
# Figure 2
# #######################################
# parameters from methods section
eta = 1e3 # g / cm /s

# lengths in cm
lengths = np.array([5.2, 7.8, 9.4, 3.1, 7.5])
wavelengths = lengths / np.array([5, 5, 5, 3, 7])
thetas = np.array([55, 39, 20, 55, 56]) * np.pi/180
# theta = np.arctan(2*np.pi * R / wavelen) # pitch angle
R = np.tan(thetas) * wavelengths / (2*np.pi)
A = np.array([0.67, 0.71, 0.74, 0.48, 0.91]) * 6*np.pi
B = np.array([0.032, 0.038, 0.018, 0.023, 0.053]) * 6*np.pi
D = np.array([0.076, 0.06, 0.031, 0.053, 0.13]) * 6*np.pi

# A / (eta * L)
a_stars = A / lengths
a_star_unc = np.std(a_stars)
# D / (eta * L * D^2)
d_stars = D / (lengths * R**2)
d_star_unc = np.std(d_stars)
# B / (eta * L * D)
b_stars = B / (lengths * R)
b_star_unc = np.std(b_stars)

effs = b_stars**2 / (4 * a_stars * d_stars)

print("averaged experimental values")
print(f"B*={np.mean(b_stars):.3f}({b_star_unc * 1e3:.0f})")
print(f"D*={np.mean(d_stars):.3f}({d_star_unc * 1e3:.0f})")
print(f"A*={np.mean(a_stars):.3f}({a_star_unc * 1e3:.0f})")
