"""
Extracted data from https://www.pnas.org/doi/epdf/10.1073/pnas.1219831110 using https://apps.automeris.io/wpd/
"""

import numpy as np

# #######################################
# Figure 2
# #######################################
# parameters from methods section
eta = 1e2 # kg / m / s
R = 6.3e-3 # m
Diam = 2 * R
L = 20 * R
a = R / 16 # m
# theta = np.arctan(2*np.pi * R / wavelen) # pitch angle

# length in m. Values in plot are L / lambda
wavelengths = np.array([2.226, 3.677, 6.509, 9.811, 15.241]) * R
thetas = np.arctan(2*np.pi * R / wavelengths)

# thrust / (eta * omega * R^2) = A_{12} / (eta * R^2)
thrust = np.array([6.264367816091955, 9.597701149425284, 11.034482758620687, 9.568965517241377, 7.844827586206897])
# torque / (eta * omega * R^3) = A_{22} / (eta * R^3)
torque = np.array([131.92771084337352, 100.80321285140562, 81.92771084337349, 72.69076305220884, 68.07228915662648])
# drag / (eta * U * R) = A_{11} / (eta * R)
drag = np.array([52.05882352941176, 46.96078431372548, 39.41176470588235, 35.29411764705881, 34.11764705882352])


# A / (eta * L)
a_stars = drag * R / L
a_star_unc = np.std(drag * R / L)
# D / (eta * L * D^2)
d_stars = torque * R**3 / (L * R ** 2)
d_star_unc = np.std(torque * R ** 3 / (L * R ** 2))
# B / (eta * L * D)
b_stars = thrust * R**2 / (L * R)
b_star_unc = np.std(thrust * R ** 2 / (L * R))

print("averaged experimental values")
print(f"B*={np.mean(b_stars):.3f}({b_star_unc * 1e3:.0f})")
print(f"D*={np.mean(d_stars):.3f}({d_star_unc * 1e3:.0f})")
print(f"A*={np.mean(a_stars):.3f}({a_star_unc * 1e3:.0f})")


