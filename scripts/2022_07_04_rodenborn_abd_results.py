"""
print ABD results from Rodenborn et al https://www.pnas.org/doi/epdf/10.1073/pnas.1219831110
figure 2
"""
import numpy as np

# as functions of cn and ct, the RFT drag coefficients
def A(cn, ct, l, theta): return (cn * np.sin(theta)**2 + ct * np.cos(theta)**2) * l / np.cos(theta)
def D(cn, ct, l, R, theta): return R**2 * (cn * np.cos(theta)**2 + ct * np.sin(theta)**2) * l / np.cos(theta)
def B(cn, ct, l, R, theta): return R * (cn - ct) * np.sin(theta) * np.cos(theta) * l / np.cos(theta)

# Hancocks drag coefficients (divided by viscosity)
def ct_h(wavelen_div_a): return 2*np.pi / (np.log(2* wavelen_div_a) - 0.5)
def cn_h(wavelen_div_a): return 4*np.pi / (np.log(2* wavelen_div_a) + 0.5)

# Lighthill's drag coefficients
def ct_l(wavelen_div_a, theta): return 2*np.pi / (np.log(0.18*wavelen_div_a / np.cos(theta)))
def cn_l(wavelen_div_a, theta): return 4*np.pi / (np.log(0.18*wavelen_div_a / np.cos(theta)) + 0.5)

# #######################################
# Rodenborn Figure 2
# #######################################
# parameters from methods section
eta_rodenborn = 1e2 # kg / m / s
R = 6.3e-3 # m
L = 130e-3
a = 0.397e-3

# length in m. Values in plot are L / lambda
wavelengths = np.array([2.226, 3.677, 6.509, 9.811, 15.241]) * R
thetas = np.arctan(2 * np.pi * R / wavelengths)

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
d_stars = torque * R ** 3 / (L * R ** 2)
# B / (eta * L * D)
b_stars = thrust * R ** 2 / (L * R)

effs = b_stars**2 / (4 * a_stars * d_stars)

print("Rodenborn figure 2 values")
print(f"R = {R * 1e3:.2f}mm, L = {L * 1e3:.2f}mm, a = {a * 1e3:.3f}mm")
print(f"L/R = {L/R:.6f}")
print(f"a/R = {a/R:.6f}")

print(f"wlen (mm){'':<5}, ", end="")
print(f"wlen/a{'':<5}, ", end="")
print(f"th (deg){'':<5},", end="")
print(f"B*{'':<10}, ", end="")
print(f"D*{'':<10}, ", end="")
print(f"A*{'':<10}, ", end="")
print(f"eff max")
for ii in range(len(wavelengths)):
    print(f"{wavelengths[ii] * 1e3:<15.2f}", end="")
    print(f"{wavelengths[ii] / a:<15.2f}", end="")
    print(f"{thetas[ii] * 180 / np.pi:<15.2f}", end="")
    print(f"{b_stars[ii]:<15.2f}", end="")
    print(f"{d_stars[ii]:<15.2f}", end="")
    print(f"{a_stars[ii]:<15.2f}", end="")
    print(f"{effs[ii]*1e2:<15.2f}%")

# #############################
# Rodenborn et al figure 2, fourth data point (fourth longest wavelength), with theta ~32 degrees
# compute ABD values
# #############################
index = 3

cols = ["B /eta*R**2", "D /eta*R**3", "A /eta*R"]
rows = ["expt", "lighthill sbt", "johnson sbt", "stokeslets", "GH rft", "lighthill rft"]
bda_mat = np.array([[thrust[index], torque[index], drag[index]],
                    [9.251, 69.27, 34.13],
                    [9.156, 66.43, 32.88],
                    [np.nan, np.nan, np.nan],
                    [B(cn_h(wavelengths[index] / a), ct_h(wavelengths[index] / a), L, R, thetas[index]) / R ** 2,
                     D(cn_h(wavelengths[index] / a), ct_h(wavelengths[index] / a), L, R, thetas[index]) / R ** 3,
                     A(cn_h(wavelengths[index] / a), ct_h(wavelengths[index] / a), L, thetas[index]) / R],
                    [B(cn_l(wavelengths[index] / a, thetas[index]), ct_l(wavelengths[index] / a, thetas[index]), L, R, thetas[index]) / R ** 2,
                     D(cn_l(wavelengths[index] / a, thetas[index]), ct_l(wavelengths[index] / a, thetas[index]), L, R, thetas[index]) / R ** 3,
                     A(cn_l(wavelengths[index] / a, thetas[index]), ct_l(wavelengths[index] / a, thetas[index]), L, thetas[index]) / R],
                    ])

# convert to our normalization
cols_star = ["B /eta*L*R", "D /eta*L*R**2", "A /eta*L"]
bda_star_mat = np.array(bda_mat, copy=True)
bda_star_mat[:, 0] = bda_star_mat[:, 0] * R**2 / (L * R)
bda_star_mat[:, 1] = bda_star_mat[:, 1] * R**3 / (L * R ** 2)
bda_star_mat[:, 2] = bda_star_mat[:, 2] * R / (L)


# print results


# print our normalization
print(f"{'':<20}, ", end="")
for ii in range(3):
    print(f"{cols_star[ii]:<20}, ", end="")
print("")

for ii in range(len(rows)):
    print(f"{rows[ii]:<20}, ", end="")
    for jj in range(3):
        print(f"{bda_star_mat[ii, jj]:<20.3f}, ", end="")
    print("")

# print Rodenborn form
print(f"{'':<20}, ", end="")
for ii in range(3):
    print(f"{cols[ii]:<20}, ", end="")
print("")

for ii in range(len(rows)):
    print(f"{rows[ii]:<20}, ", end="")
    for jj in range(3):
        print(f"{bda_mat[ii, jj]:<20.3f}, ", end="")
    print("")
