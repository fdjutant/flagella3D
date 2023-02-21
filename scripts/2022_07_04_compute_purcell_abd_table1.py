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

print(f"radius (cm){'':<5}, ", end="")
print(f"wlen (cm){'':<5}, ", end="")
print(f"lens{'':<5}, ", end="")
print(f"th (deg){'':<5},", end="")
print(f"B*{'':<10}, ", end="")
print(f"D*{'':<10}, ", end="")
print(f"A*{'':<10}, ", end="")
print(f"eff max")
for ii in range(len(wavelengths)):
    print(f"{R[ii]:<15.2f}", end="")
    print(f"{wavelengths[ii]:<15.2f}", end="")
    print(f"{lengths[ii]:<15.2f}", end="")
    print(f"{thetas[ii] * 180 / np.pi:<15.2f}", end="")
    print(f"{b_stars[ii]:<15.2f}", end="")
    print(f"{d_stars[ii]:<15.2f}", end="")
    print(f"{a_stars[ii]:<15.2f}", end="")
    print(f"{effs[ii]*1e2:<15.2f}%")