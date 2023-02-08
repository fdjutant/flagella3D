"""
Determine reduced ABD coefficients for the experiment

Use RFT expression given in Rodenborn et. al https://doi.org/10.1073/pnas.1219831110
For references on the helix parameters, see Turner et al https://doi.org/10.1128/JB.182.10.2793-2801.2000 and
Namba et al https://doi.org/10.1038/342648a0

"""
import numpy as np
from pathlib import Path
import zarr

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

# load experimental data
data_dir = Path(r"\\10.206.26.21\flagella_project\2022_06_20_16;04;38_processed_data")

files = list(data_dir.glob("*.zarr"))
nfiles = len(files)

kb = 1.380649e-23
T = 273 + 25

len_lims = [6, 10]
patterns = ["suc40*", "suc50*", "suc70*"]
# viscosities = [0.00177, 0.00199, 0.00284] # Pa * s # older values, accidentally used in first submission
viscosities = [0.00177, 0.00241, 0.00343] # Pa * s # corrected values after measurements in supplemental section S7
# viscosities = [0.00177, 0.0029, 0.00436] # Pa * s # "by eye" estimate of diffusion coefficients from fig. S6A
diff_mats = [[] for ii in range(len(patterns))]
for ii in range(len(patterns)):
    files = list(data_dir.glob(f"{patterns[ii]:s}.zarr"))
    for jj, f in enumerate(files):
        data = zarr.open(f)

        length = np.mean(data.lengths)
        if length < len_lims[0] or length > len_lims[1]:
            continue

        # convert from um to m
        diff_mat = np.array([[data.diffusion_constants_body[0, 0, 10] * 1e-6**2, data.diffusion_constants_body[0, 3, 10] * 1e-6],
                             [data.diffusion_constants_body[3, 0, 10] * 1e-6, data.diffusion_constants_body[3, 3, 10]]])

        diff_mats[ii].append(diff_mat)

diff_mats = [np.array(diff_mats[ii]) for ii in range(len(diff_mats))]
nflagella = np.sum(len(diff_mats[ii]) for ii in range(len(diff_mats)))
diff_mats_mean = [np.mean(diff_mats[ii], axis=0) for ii in range(len(diff_mats))]
prop_mats = [kb * T * np.linalg.inv(diff_mats_mean[ii]) for ii in range(len(diff_mats))]

# helix parameter estimates for this experiment
# R = 0.25 # helical radius
# L = 7.7 # length in um
# Diam = 2 * R
# wavelen = 2. # pitch in um
# a = 0.01 # filament radius
# theta = np.arctan(2*np.pi * R / wavelen) # pitch angle
#
# cols = ["B /eta*R**2", "D /eta*R**3", "A /eta*R"]
# cols_star = ["B /eta*L*D", "D /eta*L*D**2", "A /eta*L"]
# rows = ["expt", "lighthill sbt", "johnson sbt", "stokeslets", "GH rft", "lighthill rft"]
# # SBT/stokeslet computed using Rodenborn matlab program
# # note: in GUI units are absolute, not fractions of R as might expect output units ssame as Rodenborn
# # non-dimensionalized form however, their program has issues if R is not set to 1
# bda_mat = np.array([[0, 0, 0],
#                     [13.27, 96.27, 46.19],
#                     [13.08, 92.64, 44.78],
#                     [13.178, 95.797, 45.917],
#                     [B(cn_h(wavelen / a), ct_h(wavelen / a), L, R, theta) / R ** 2,
#                      D(cn_h(wavelen / a), ct_h(wavelen / a), L, R, theta) / R ** 3,
#                      A(cn_h(wavelen / a), ct_h(wavelen / a), L, theta) / R],
#                     [B(cn_l(wavelen / a, theta), ct_l(wavelen / a, theta), L, R, theta) / R ** 2,
#                      D(cn_l(wavelen / a, theta), ct_l(wavelen / a, theta), L, R, theta) / R ** 3,
#                      A(cn_l(wavelen / a, theta), ct_l(wavelen / a, theta), L, theta) / R],
#                     ])



# helix parameter estimates for this experiment
# R = 0.17 # helical radius
# L = 7.7 # length in um
# Diam = 2 * R
# wavelen = 2.5 # pitch in um
# a = 0.01 # filament radius
# theta = np.arctan(2*np.pi * R / wavelen) # pitch angle
#
# cols = ["B /eta*R**2", "D /eta*R**3", "A /eta*R"]
# cols_star = ["B /eta*L*D", "D /eta*L*D**2", "A /eta*L"]
# rows = ["expt", "lighthill sbt", "johnson sbt", "stokeslets", "GH rft", "lighthill rft"]
# # SBT/stokeslet computed using Rodenborn matlab program
# # note: in GUI units are absolute, not fractions of R as might expect output units ssame as Rodenborn
# # non-dimensionalized form however, their program has issues if R is not set to 1
# bda_mat = np.array([[0, 0, 0],
#                     [13.8096, 134.465, 56.8],
#                     [13.6121, 129.656, 54.866],
#                     [0, 0, 0],
#                     [B(cn_h(wavelen / a), ct_h(wavelen / a), L, R, theta) / R ** 2,
#                      D(cn_h(wavelen / a), ct_h(wavelen / a), L, R, theta) / R ** 3,
#                      A(cn_h(wavelen / a), ct_h(wavelen / a), L, theta) / R],
#                     [B(cn_l(wavelen / a, theta), ct_l(wavelen / a, theta), L, R, theta) / R ** 2,
#                      D(cn_l(wavelen / a, theta), ct_l(wavelen / a, theta), L, R, theta) / R ** 3,
#                      A(cn_l(wavelen / a, theta), ct_l(wavelen / a, theta), L, theta) / R],
#                     ])


R = 0.25 # helical radius
L = 7.7 # length in um
Diam = 2 * R
wavelen = 2.5 # pitch in um
a = 0.01 # filament radius
theta = np.arctan(2*np.pi * R / wavelen) # pitch angle

b_exp = [prop_mats[ii][1, 0] / viscosities[ii] / (R * 1e-6)**2 for ii in range(len(prop_mats))]
d_exp = [prop_mats[ii][1, 1] / viscosities[ii] / (R * 1e-6)**3 for ii in range(len(prop_mats))]
a_exp = [prop_mats[ii][0, 0] / viscosities[ii] / (R * 1e-6) for ii in range(len(prop_mats))]

cols = ["B /eta*R**2", "D /eta*R**3", "A /eta*R"]
cols_star = ["B /eta*L*R", "D /eta*L*R**2", "A /eta*L"]
rows = ["expt average",
        f"expt eta={viscosities[0] * 1e3:.2f} mPa*s",
        f"expt eta={viscosities[1] * 1e3:.2f} mPa*s",
        f"expt eta={viscosities[2] * 1e3:.2f} mPa*s",
        "lighthill sbt", "johnson sbt", "stokeslets", "GH rft", "lighthill rft"]
# SBT/stokeslet computed using Rodenborn matlab program
# note: in GUI units are absolute, not fractions of R as might expect output units ssame as Rodenborn
# non-dimensionalized form however, their program has issues if R is not set to 1
bda_mat = np.array([[np.mean(b_exp), np.mean(d_exp), np.mean(a_exp)],
                    [b_exp[0], d_exp[0], a_exp[0]],
                    [b_exp[1], d_exp[1], a_exp[1]],
                    [b_exp[2], d_exp[2], a_exp[2]],
                    [12.1864, 91.1528, 43.0764],
                    [12.0108, 87.8737, 41.7128],
                    [12.0935, 90.8023, 42.810],
                    [B(cn_h(wavelen / a), ct_h(wavelen / a), L, R, theta) / R ** 2,
                     D(cn_h(wavelen / a), ct_h(wavelen / a), L, R, theta) / R ** 3,
                     A(cn_h(wavelen / a), ct_h(wavelen / a), L, theta) / R],
                    [B(cn_l(wavelen / a, theta), ct_l(wavelen / a, theta), L, R, theta) / R ** 2,
                     D(cn_l(wavelen / a, theta), ct_l(wavelen / a, theta), L, R, theta) / R ** 3,
                     A(cn_l(wavelen / a, theta), ct_l(wavelen / a, theta), L, theta) / R],
                    ])

# convert to our normalization
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

