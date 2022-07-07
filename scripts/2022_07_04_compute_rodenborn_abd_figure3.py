"""
Extracted data from https://www.pnas.org/doi/epdf/10.1073/pnas.1219831110 using https://apps.automeris.io/wpd/
"""

import numpy as np

# #######################################
# Figure 3
# #######################################
# parameters from methods section
eta = 1e2 # kg / m / s
R = 6.6e-3 # m
Diam = 2 * R
wavelen = 2.42 * R # m
a = R / 16 # m
theta = np.arctan(2*np.pi * R / wavelen) # pitch angle

# length in m. Values in plot are L / lambda
lengths = np.array([3.4, 5.75, 7.3, 9.2, 11.25]) * wavelen
# thrust / (eta * omega * R^2) = A_{12} / (eta * R^2)
thrust = np.array([3.2799245994344943, 4.665409990574929, 6.0367577756833235, 7.6908576814326075, 8.934967012252589])
# torque / (eta * omega * R^3) = A_{22} / (eta * R^3)
torque = np.array([54.33153293718922, 81.70438767075055, 108.08735670146925, 137.78734445839584, 171.74241712435827])
# drag / (eta * U * R) = A_{11} / (eta * R)
drag = np.array([26.61290322580645, 37.96526054590571, 46.153846153846146, 52.85359801488833, 59.83250620347394])

# A / (eta * L)
a_stars = drag * R / lengths
a_star_unc = np.std(drag * R / lengths)
# D / (eta * L * D^2)
d_stars = torque * R**3 / (lengths * R ** 2)
d_star_unc = np.std(torque * R ** 3 / (lengths * R ** 2))
# B / (eta * L * D)
b_stars = thrust * R**2 / (lengths * R)
b_star_unc = np.std(thrust * R ** 2 / (lengths * R))

print("averaged experimental values")
print(f"B*={np.mean(b_stars):.3f}({b_star_unc * 1e3:.0f})")
print(f"D*={np.mean(d_stars):.3f}({d_star_unc * 1e3:.0f})")
print(f"A*={np.mean(a_stars):.3f}({a_star_unc * 1e3:.0f})")


