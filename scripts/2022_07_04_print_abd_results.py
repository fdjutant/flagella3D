"""
print results
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

# #############################
# Rodenborn et al figure 3, longest flagella
# #############################
# R = 6.6e-3 # m
# L = 0.179685
# Diam = 2 * R
# wavelen = 2.42 * R # m
# a = R / 16 # m
# theta = np.arctan(2*np.pi * R / wavelen) # pitch angle
#
# cols = ["B /eta*R**2", "D /eta*R**3", "A /eta*R"]
# rows = ["expt", "lighthill sbt", "johnson sbt", "stokeslets", "GH rft", "lighthill rft"]
# cols_star = ["B /eta*L*R", "D /eta*L*R**2", "A /eta*L"]
# bda_mat = np.array([[8.934967012252589, 171.74241712435827, 59.83250620347394],
#                     [7.8096, 170.309, 62.29],
#                     [7.807, 161.984, 60.74],
#                     [7.758, 169.931, 62.136],
#                     [B(cn_h(wavelen / a), ct_h(wavelen / a), L, R, theta) / R ** 2,
#                      D(cn_h(wavelen / a), ct_h(wavelen / a), L, R, theta) / R ** 3,
#                      A(cn_h(wavelen / a), ct_h(wavelen / a), L, theta) / R],
#                     [B(cn_l(wavelen / a, theta), ct_l(wavelen / a, theta), L, R, theta) / R ** 2,
#                      D(cn_l(wavelen / a, theta), ct_l(wavelen / a, theta), L, R, theta) / R ** 3,
#                      A(cn_l(wavelen / a, theta), ct_l(wavelen / a, theta), L, theta) / R],
#                     ])

# #############################
# Rodenborn et al figure 2, fourth data point (fourth longest wavelength), with theta ~32 degrees
# #############################
R = 6.3e-3 # m
L = 129.97e-3
# Diam = 2 * R
wavelen = 9.811 * R # m
a = 0.063 * R # m
theta = np.arctan(2*np.pi * R / wavelen) # pitch angle

cols = ["B /eta*R**2", "D /eta*R**3", "A /eta*R"]
rows = ["expt", "lighthill sbt", "johnson sbt", "stokeslets", "GH rft", "lighthill rft"]
bda_mat = np.array([[9.568965517241377, 72.69076305220884, 35.29411764705881],
                    [9.251, 69.27, 34.13],
                    [9.156, 66.43, 32.88],
                    [0, 0, 0],
                    [B(cn_h(wavelen / a), ct_h(wavelen / a), L, R, theta) / R ** 2,
                     D(cn_h(wavelen / a), ct_h(wavelen / a), L, R, theta) / R ** 3,
                     A(cn_h(wavelen / a), ct_h(wavelen / a), L, theta) / R],
                    [B(cn_l(wavelen / a, theta), ct_l(wavelen / a, theta), L, R, theta) / R ** 2,
                     D(cn_l(wavelen / a, theta), ct_l(wavelen / a, theta), L, R, theta) / R ** 3,
                     A(cn_l(wavelen / a, theta), ct_l(wavelen / a, theta), L, theta) / R],
                    ])


# #############################
# between rodenborn figure 3 and 4. Start with fig 3 value and change only wavelength
# #############################
# R = 6.6e-3 # m
# L = 27.225 * R
# wavelen = 10 * R # m
# a = 0.0625 * R # m
# theta = np.arctan(2*np.pi * R / wavelen) # pitch angle
#
# cols = ["B /eta*R**2", "D /eta*R**3", "A /eta*R"]
# rows = ["johnson sbt", "GH rft", "lighthill rft"]
# bda_mat = np.array([[11.3106, 86.7, 40.46],
#                     [B(cn_h(wavelen / a), ct_h(wavelen / a), L, R, theta) / R ** 2,
#                      D(cn_h(wavelen / a), ct_h(wavelen / a), L, R, theta) / R ** 3,
#                      A(cn_h(wavelen / a), ct_h(wavelen / a), L, theta) / R],
#                     [B(cn_l(wavelen / a, theta), ct_l(wavelen / a, theta), L, R, theta) / R ** 2,
#                      D(cn_l(wavelen / a, theta), ct_l(wavelen / a, theta), L, R, theta) / R ** 3,
#                      A(cn_l(wavelen / a, theta), ct_l(wavelen / a, theta), L, theta) / R],
#                     ])

# #############################
# between rodenborn figure 3 and 4. Start with previous values, keep helical pitch angle constant but now change
# length
# #############################
# R = 6.6e-3 # m
# L = 742.5e-3
# wavelen = 10 * R # m
# a = 0.0625 * R # m
# theta = np.arctan(2*np.pi * R / wavelen) # pitch angle
#
# cols = ["B /eta*R**2", "D /eta*R**3", "A /eta*R"]
# rows = ["johnson sbt", "GH rft", "lighthill rft"]
# bda_mat = np.array([[35.924,354.039, 126.289],
#                     [B(cn_h(wavelen / a), ct_h(wavelen / a), L, R, theta) / R ** 2,
#                      D(cn_h(wavelen / a), ct_h(wavelen / a), L, R, theta) / R ** 3,
#                      A(cn_h(wavelen / a), ct_h(wavelen / a), L, theta) / R],
#                     [B(cn_l(wavelen / a, theta), ct_l(wavelen / a, theta), L, R, theta) / R ** 2,
#                      D(cn_l(wavelen / a, theta), ct_l(wavelen / a, theta), L, R, theta) / R ** 3,
#                      A(cn_l(wavelen / a, theta), ct_l(wavelen / a, theta), L, theta) / R],
#                     ])

# #############################
# between rodenborn figure 3 expt
# #############################
# R = 0.25 # um
# L = 7.7
# wavelen = 2.5 # m
# a = 0.0625 * R # m
# theta = np.arctan(2*np.pi * R / wavelen) # pitch angle
#
# cols = ["B /eta*R**2", "D /eta*R**3", "A /eta*R"]
# rows = ["johnson sbt", "GH rft", "lighthill rft"]
# bda_mat = np.array([[12.495, 97.9743, 44.553],
#                     [B(cn_h(wavelen / a), ct_h(wavelen / a), L, R, theta) / R ** 2,
#                      D(cn_h(wavelen / a), ct_h(wavelen / a), L, R, theta) / R ** 3,
#                      A(cn_h(wavelen / a), ct_h(wavelen / a), L, theta) / R],
#                     [B(cn_l(wavelen / a, theta), ct_l(wavelen / a, theta), L, R, theta) / R ** 2,
#                      D(cn_l(wavelen / a, theta), ct_l(wavelen / a, theta), L, R, theta) / R ** 3,
#                      A(cn_l(wavelen / a, theta), ct_l(wavelen / a, theta), L, theta) / R],
#                     ])

# convert to our normalization
cols_star = ["B /eta*L*R", "D /eta*L*R**2", "A /eta*L"]
bda_star_mat = np.array(bda_mat, copy=True)
bda_star_mat[:, 0] = bda_star_mat[:, 0] * R**2 / (L * R)
bda_star_mat[:, 1] = bda_star_mat[:, 1] * R**3 / (L * R ** 2)
bda_star_mat[:, 2] = bda_star_mat[:, 2] * R / (L)


# print results
print(f"R = {R:.6f}")
print(f"L = {L:.6f}")
print(f"a = {a:.6f}")
print(f"wlen = {wavelen:.6f}")
print(f"L/R = {L/R:.6f}")
print(f"a/R = {a/R:.6f}")
print(f"wlen/R = {wavelen/R:.6f}")
print(f"wlen/a = {wavelen/a:.6f}")
print(f"theta = {theta * 180/np.pi:.6f}deg")

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
