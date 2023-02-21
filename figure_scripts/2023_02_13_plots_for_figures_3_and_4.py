"""
Load flagella diffusion data analysis and generate all plots in the paper using it, as well as other numbers
such as Flagella length, ratios of diffusion coefficients, and ABD coefficients which appear in the paper
"""
# %% Import all necessary libraries
import numpy as np
import matplotlib
import scipy.interpolate

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import datetime
from pathlib import Path
import zarr
from scipy import optimize
from scipy.optimize import least_squares
from scipy.stats import sem

# time settings in the light sheet
pxum = 0.115
camExposure_ms = 2
sweep_um = 15
stepsize_nm = 400
exp3D_sec = 1e-3 * camExposure_ms * (sweep_um * 1e3 / stepsize_nm)
kB = 1.380649e-23  # J / K
T = 273 + 25  # K
min_length_um = 6
max_length_um = 10
npts_msd_fit = 10

# non-measured helix parameters
R = 0.25 # helical radius um
a = 0.01 # filament radius um

save_plots = False

# ##################################
# load results
# ##################################
tstamp = datetime.datetime.now().strftime('%Y_%m_%d_%H;%M;%S')

data_folder = Path(r"\\10.206.26.21\flagella_project\2022_06_20_16;04;38_processed_data")
save_dir = Path(r"\\10.206.26.21\flagella_project\various-plots")

patterns = ["suc40*.zarr",
            "suc50*.zarr",
            "suc70*.zarr"]

# viscosity for 40% sucrose mixture is weighted average of two measurements
v40s = np.array([1.63, 2.4])
v40s_unc = np.array([0.03, 0.08])
v40 = np.sum(v40s / v40s_unc) / np.sum(1 / v40s_unc)
v40unc = 1 / np.sqrt(np.sum(1 / v40s_unc))

initial_sucrose_wv = np.array([0.4, 0.5, 0.7])
viscosities = np.array([np.round(v40, 2), 3.0, 4.56]) # mPa * s
viscosities_unc = np.array([np.round(v40unc, 2), 0.2, 0.14])


diff_coeffs = []
lens = []
flagella_wlens = []
msds = []
lag_times = []
msds_fps = []
for p in patterns:
    files = list(data_folder.glob(p))

    diff_coeffs_visc = []
    lag_times_visc = []
    lens_visc = []
    msds_visc = []
    msd_fps_visc = []
    flagella_wlens_visc = []
    for f in files:
        z = zarr.open(f)

        lens_visc.append(np.mean(z["lengths"]))
        flagella_wlens_visc.append(2*np.pi / np.mean(z["helix_fit_params"][:, 1]) * pxum)
        diff_coeffs_visc.append(z["diffusion_constants_body"][:, :, npts_msd_fit])
        msds_visc.append(np.array(z["msds_body"]))
        lag_times_visc.append(np.array(z["lag_times"]))
        msd_fps_visc.append(np.array(z["msd_fit_params_body"]))


    diff_coeffs.append(np.stack(diff_coeffs_visc, axis=0))
    lag_times.append(lag_times_visc)
    flagella_wlens.append(np.stack(flagella_wlens_visc))
    lens.append(np.stack(lens_visc))
    msds.append(msds_visc)
    msds_fps.append(msd_fps_visc)

# ##################################
# collate results
# ##################################
jitter = 0.08

eta_pts = np.concatenate([np.ones(len(d)) * e + np.random.uniform(-jitter, jitter, len(d)) for e, d in zip(viscosities, diff_coeffs)])
ds = np.concatenate(diff_coeffs, axis=0)
to_use = [np.logical_and(l >= min_length_um, l <= max_length_um) for l in lens]

dmeans = np.stack([np.mean(d[use], axis=0) for d, use in zip(diff_coeffs, to_use)], axis=0)
duncs = np.stack([sem(d[use], axis=0) for d, use in zip(diff_coeffs, to_use)], axis=0)

# translational diffusion coefficients
d_par = dmeans[:, 0, 0]
d_par_unc = duncs[:, 0, 0]

d_perp = [np.mean(np.concatenate((d[use, 1, 1], d[use, 2, 2]))) for d, use in zip(diff_coeffs, to_use)]
d_perp_unc = [sem(np.concatenate((d[use, 1, 1], d[use, 2, 2]))) for d, use in zip(diff_coeffs, to_use)]

d_translation_ratios = np.array([par / perp for perp, par in zip(d_perp, d_par)])
d_translation_ratios_unc = np.array([np.sqrt((perp_err/perp)**2 + (par_err/par)**2) for
                                     perp, perp_err, par, par_err in zip(d_perp, d_perp_unc, d_par, d_par_unc)])

# rotational diffusion coefficients
d_roll = dmeans[:, 3, 3]
d_roll_unc = duncs[:, 3, 3]

d_yaw = [np.mean(np.concatenate((d[use, 4, 4], d[use, 5, 5]))) for d, use in zip(diff_coeffs, to_use)]
d_yaw_unc = [sem(np.concatenate((d[use, 4, 4], d[use, 5, 5]))) for d, use in zip(diff_coeffs, to_use)]

# coupled
d_coupled = dmeans[:, 0, 3]
d_coupled_unc = duncs[:, 0, 3]

#
lens_arr = np.concatenate(lens)
wlen_arr = np.concatenate(flagella_wlens)
to_use_arr = np.concatenate(to_use)

# lengths
L = np.mean(lens_arr[to_use_arr])
std_len = np.std(lens_arr[to_use_arr])
mean_len_exps = np.array([np.mean(l[use]) for l, use in zip(lens, to_use)])
std_lens_exps = np.array([np.std(l[use]) for l, use in zip(lens, to_use)])

wavelen = np.mean(wlen_arr[to_use_arr])
wlen_unc = np.std(wlen_arr[to_use_arr])

n_expts = np.array([np.sum(use) for use in to_use])

# ##################################
# compute non-dimensionalized propulsion matrix
# ##################################
diff_mats_2d_mean = [np.array([[d_par[ii] * 1e-6**2, d_coupled[ii] * 1e-6],
                               [d_coupled[ii] * 1e-6, d_roll[ii]]]) for ii in range(len(viscosities))]
prop_mats = [kB * T * np.linalg.inv(diff_mats_2d_mean[ii]) for ii in range(len(viscosities))]

b_exp = [prop_mats[ii][1, 0] / (viscosities[ii] * 1e-3) / (R * 1e-6) / (mean_len_exps[ii] * 1e-6) for ii in range(len(prop_mats))]
d_exp = [prop_mats[ii][1, 1] / (viscosities[ii] * 1e-3) / (R * 1e-6)**2 / (mean_len_exps[ii] * 1e-6) for ii in range(len(prop_mats))]
a_exp = [prop_mats[ii][0, 0] / (viscosities[ii] * 1e-3) / (mean_len_exps[ii] * 1e-6) for ii in range(len(prop_mats))]

b_exp_unc = [b_exp[ii] * np.sqrt((viscosities_unc[ii] / viscosities[ii])**2 + (std_lens_exps[ii] / mean_len_exps[ii])**2) for ii in range(len(viscosities))]
d_exp_unc = [d_exp[ii] * np.sqrt((viscosities_unc[ii] / viscosities[ii])**2 + (std_lens_exps[ii] / mean_len_exps[ii])**2) for ii in range(len(viscosities))]
a_exp_unc = [a_exp[ii] * np.sqrt((viscosities_unc[ii] / viscosities[ii])**2 + (std_lens_exps[ii] / mean_len_exps[ii])**2) for ii in range(len(viscosities))]

b_exp_r = [b_exp[ii] * mean_len_exps[ii] / R for ii in range(len(prop_mats))]
d_exp_r = [d_exp[ii] * mean_len_exps[ii] / R for ii in range(len(prop_mats))]
a_exp_r = [a_exp[ii] * mean_len_exps[ii] / R for ii in range(len(prop_mats))]

eff_max = [b_exp[ii]**2 / (4 * a_exp[ii] * d_exp[ii]) for ii in range(len(viscosities))]
eff_max_unc = [np.sqrt((2 * b_exp[ii] * b_exp_unc[ii] / (4 * a_exp[ii] * d_exp[ii]))**2 +
                       (b_exp[ii] ** 2 / (4 * a_exp[ii]**2 * d_exp[ii]) * a_exp_unc[ii])**2 +
                       (b_exp[ii] ** 2 / (4 * a_exp[ii] * d_exp[ii]**2) * d_exp_unc[ii])**2) for ii in range(len(viscosities))]

eff_max_avg = np.mean(eff_max)
eff_max_avg_unc = np.sqrt(np.sum(np.array(eff_max_unc)**2)) / len(eff_max)

print("efficiencies:")
for ii in range(len(viscosities)):
    print(f"{eff_max[ii]*1e2:.3f} +/- {eff_max_unc[ii] * 1e2:.3f}")

print("mean efficiency")
print(f"{eff_max_avg*1e2:.3f} +/- {eff_max_avg_unc * 1e2:.3f}")

# ##################################
# fit lines versus viscosity
# ##################################

def inv_line(x, p): return p[0] / x

r_xlation_dparallel = least_squares(lambda p: inv_line(viscosities, p) - d_par, [1])
r_xlation_dperp = least_squares(lambda p: inv_line(viscosities, p) - d_perp, [1])
r_droll = least_squares(lambda p: inv_line(viscosities, p) - d_roll, [1])
r_dyaw = least_squares(lambda p: inv_line(viscosities, p) - d_yaw, [1])
r_droll_dparallel = least_squares(lambda p: inv_line(viscosities, p) - d_coupled, [1])

interp_etas = np.linspace(np.min(viscosities), np.max(viscosities), 1001)

interp_times = np.linspace(0, lag_times[0][0][:npts_msd_fit], 1000)

# ##################################
# Figure 3
# ##################################
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 12})

figh = plt.figure(dpi=300, figsize=(10, 6.2))
grid = figh.add_gridspec(nrows=2, hspace=0.4,
                         ncols=3, wspace=0.4)

# translational MSD
index = 6
visc_index = 2


ax = figh.add_subplot(grid[0, 0])
ax.set_xlabel("Lag time [s]")
ax.set_ylabel("MSD [$\mu m^2$]")
ax.plot(lag_times[visc_index][index][:2*npts_msd_fit], msds[visc_index][index][0, 0, :2*npts_msd_fit],
        "o", color="thistle", mew=1, mec="purple",
        label=r"$\langle \Delta_{n_1}^2 \rangle$")
ax.plot(interp_times, interp_times * msds_fps[visc_index][index][0, 0, npts_msd_fit, 0] +
                                     msds_fps[visc_index][index][0, 0, npts_msd_fit, 1],
        color="purple")
ax.plot(lag_times[visc_index][index][:2*npts_msd_fit], msds[visc_index][index][1, 1, :2*npts_msd_fit],
        "o", color="lightgoldenrodyellow", mew=1, mec="orange",
        label=r"$\langle \Delta_{n_2}^2 \rangle$")
ax.plot(interp_times, interp_times * msds_fps[visc_index][index][1, 1, npts_msd_fit, 0] +
                                     msds_fps[visc_index][index][1, 1, npts_msd_fit, 1],
        color="orange")
ax.plot(lag_times[visc_index][index][:2*npts_msd_fit],
        msds[visc_index][index][2, 2, :2*npts_msd_fit],
        "o", color="springgreen", mew=1, mec="green",
        label=r"$\langle \Delta_{n_3}^2 \rangle$")
ax.plot(interp_times, interp_times * msds_fps[visc_index][index][2, 2, npts_msd_fit, 0] +
                                     msds_fps[visc_index][index][2, 2, npts_msd_fit, 1],
        color="green")
ax.legend(loc="upper left",
          frameon=False,
          markerfirst=True,
          fontsize=10, columnspacing=4)

# roll MSAD
ax = figh.add_subplot(grid[0, 1])
ax.set_xlabel("Lag time [s]")
ax.set_ylabel("MSAD [rad$^2$]")

ax.plot(lag_times[visc_index][index][:2*npts_msd_fit], msds[visc_index][index][3, 3, :2*npts_msd_fit],
        "o", color="thistle", mew=1, mec="purple",
        label=r"$\langle \Delta_{\psi_1}^2 \rangle$")
ax.plot(interp_times, interp_times * msds_fps[visc_index][index][3, 3, npts_msd_fit, 0] +
                                     msds_fps[visc_index][index][3, 3, npts_msd_fit, 1],
        color="purple")

ax.legend(loc="upper left",
          frameon=False,
          markerfirst=True,
          fontsize=10, columnspacing=4)

# pitch/yaw MSAD
ax = figh.add_subplot(grid[0, 2])
ax.set_xlabel("Lag time [s]")
ax.set_ylabel("MSAD [rad$^2$]")

ax.plot(lag_times[visc_index][index][:2*npts_msd_fit], msds[visc_index][index][4, 4, :2*npts_msd_fit],
        "o", color="lightgoldenrodyellow", mew=1, mec="orange",
        label=r"$\langle \Delta_{\psi_2}^2 \rangle$")
ax.plot(interp_times, interp_times * msds_fps[visc_index][index][4, 4, npts_msd_fit, 0] +
                                     msds_fps[visc_index][index][4, 4, npts_msd_fit, 1],
        color="orange")
ax.plot(lag_times[visc_index][index][:2*npts_msd_fit], msds[visc_index][index][5, 5, :2*npts_msd_fit],
        "o", color="springgreen", mew=1, mec="green",
        label=r"$\langle \Delta_{\psi_3}^2 \rangle$")
ax.plot(interp_times, interp_times * msds_fps[visc_index][index][5, 5, npts_msd_fit, 0] +
                                     msds_fps[visc_index][index][5, 5, npts_msd_fit, 1],
        color="green")
ax.legend(loc="upper left",
          frameon=False,
          markerfirst=True,
          fontsize=10, columnspacing=4)

# translational diffusion
ax = figh.add_subplot(grid[1, 0])
ax.set_xlabel("Viscosity [mPa $\cdot$ s]")
ax.set_ylabel("$D$ [$\mu m^2$/s]")
ax.plot(eta_pts[to_use_arr], ds[to_use_arr, 0, 0], "o", mew=0, alpha=0.25, color="purple", label="$D_{n_1}$")
ax.errorbar(viscosities, d_par, yerr=d_par_unc, xerr=viscosities_unc, color="purple", marker='x', linestyle="none")

ax.plot(eta_pts[to_use_arr], ds[to_use_arr, 1, 1], "o", mew=0, alpha=0.25, color="orange", label="$D_{n_2}$")
ax.plot(eta_pts[to_use_arr], ds[to_use_arr, 2, 2], "o", mew=0, alpha=0.25, color="green", label="$D_{n_3}$")

ax.errorbar(viscosities, d_perp, yerr=d_perp_unc, xerr=viscosities_unc, color="mediumturquoise", marker='x', linestyle="none")

ax.plot(interp_etas, inv_line(interp_etas, r_xlation_dparallel["x"]), color="purple")
ax.plot(interp_etas, inv_line(interp_etas, r_xlation_dperp["x"]), color="mediumturquoise")

ax.set_xticks(viscosities)
ax.set_xticklabels(viscosities)
ax.set_yticks([0, 0.1, 0.2])
ax.legend(loc="upper right", frameon=False,
          markerfirst=False,
          fontsize=10, columnspacing=4)

# roll diffusion
ax = figh.add_subplot(grid[1, 1])
ax.set_xlabel("Viscosity [mPa $\cdot$ s]")
ax.set_ylabel("$D$ [rad$^2$/s]")
ax.plot(eta_pts[to_use_arr], ds[to_use_arr, 3, 3], "o", mew=0, alpha=0.25, color="purple", label="$D_{\psi_1}$")
ax.errorbar(viscosities, d_roll, yerr=d_roll_unc, xerr=viscosities_unc, color="purple", marker='x', linestyle="none")

ax.plot(interp_etas, inv_line(interp_etas, r_droll["x"]), color="purple")

ax.set_xticks(viscosities)
ax.set_xticklabels(viscosities)
ax.set_yticks([0, 1, 2])
ax.legend(loc="upper right", frameon=False,
          markerfirst=False,
          fontsize=10, columnspacing=4)

# pitch/yaw  diffusion
ax = figh.add_subplot(grid[1, 2])
ax.set_xlabel("Viscosity [mPa $\cdot$ s]")
ax.set_ylabel("$D$ [rad$^2$/s]")
ax.plot(eta_pts[to_use_arr], ds[to_use_arr, 4, 4], "o", mew=0, alpha=0.25, color="orange", label="$D_{\psi_2}$")
ax.plot(eta_pts[to_use_arr], ds[to_use_arr, 5, 5], "o", mew=0, alpha=0.25, color="green", label="$D_{\psi_3}$")
ax.errorbar(viscosities, d_yaw, yerr=d_yaw_unc, xerr=viscosities_unc, color="mediumturquoise", marker='x', linestyle="none")

ax.plot(interp_etas, inv_line(interp_etas, r_dyaw["x"]), color="mediumturquoise")

ax.set_xticks(viscosities)
ax.set_xticklabels(viscosities)
ax.set_yticks([0, 0.025, 0.05])
ax.legend(loc="upper right", frameon=False,
          markerfirst=False,
          fontsize=10, columnspacing=4)

if save_plots:
    save_fname = save_dir / f"{tstamp:s}_fig3.pdf"
    figh.savefig(save_fname)


# ######################
# Fig 4c
# ######################
figh = plt.figure(dpi=300, figsize=(3.5, 2.5))
grid = figh.add_gridspec(nrows=1, ncols=1, bottom=0.2, left=0.25)
ax = figh.add_subplot(grid[0, 0])

ax.set_xlabel("Viscosity [mPa $\cdot$ s]")
ax.set_ylabel("$D_{n_1, \psi_1}$ [$\mu m \cdot$ rad/s]")
ax.axhline(y=0, color="black")
ax.plot(eta_pts[to_use_arr], ds[to_use_arr, 0, 3], "o", mew=0, alpha=0.25, color="gray", label="$D_{\psi_1}$")
ax.errorbar(viscosities, d_coupled, yerr=d_coupled_unc, xerr=viscosities_unc, color="black", marker='x', linestyle="none")


ax.plot(interp_etas, inv_line(interp_etas, r_droll_dparallel["x"]), color="black")

ax.set_xticks(viscosities)
ax.set_xticklabels(viscosities)
ax.set_yticks([-0.25, 0, 0.1])
# ax.legend(loc="upper right", frameon=False,
#           markerfirst=False,
#           fontsize=10, columnspacing=4)

if save_plots:
    save_fname = save_dir / f"{tstamp:s}_fig4c.pdf"
    figh.savefig(save_fname)

# ######################
# Table 4d
# ######################
theta = np.arctan(2*np.pi * R / wavelen) # pitch angle

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

# todo: fix this


cols = ["B /eta*R**2", "D /eta*R**3", "A /eta*R"]
cols_star = ["B /eta*L*R", "D /eta*L*R**2", "A /eta*L"]
rows = ["expt average",
        f"expt eta={viscosities[0]:.2f} mPa*s",
        f"expt eta={viscosities[1]:.2f} mPa*s",
        f"expt eta={viscosities[2]:.2f} mPa*s",
        "johnson sbt", "stokeslets", "GH rft", "lighthill rft"]
# SBT/stokeslet computed using Rodenborn matlab program
# note: in GUI units are absolute, not fractions of R as might expect output units same as Rodenborn
# non-dimensionalized form however, their program has issues if R is not set to 1
# [value, error, value error, ...]
bda_mat = np.array([[np.mean(b_exp_r), np.std(b_exp_r),
                     np.mean(d_exp_r), np.std(d_exp_r),
                     np.mean(a_exp_r), np.std(a_exp_r)],
                    [b_exp_r[0], np.nan, d_exp_r[0], np.nan, a_exp_r[0], np.nan],
                    [b_exp_r[1], np.nan, d_exp_r[1], np.nan, a_exp_r[1], np.nan],
                    [b_exp_r[2], np.nan, d_exp_r[2], np.nan, a_exp_r[2], np.nan],
                    # [12.1864, np.nan, 91.1528, np.nan, 43.0764, np.nan], # lighthill SBT
                    # [12.0108, np.nan, 87.8737, np.nan, 41.7128, np.nan], # Johnson SBT
                    # [12.0935, np.nan, 90.8023, np.nan,  42.810, np.nan], # stokeslets
                    [11.9275, np.nan, 87.7748, np.nan, 41.5752, np.nan], # Johnson SBT, R=1, a=0.04, wlen=10.2, len=30.9
                    [12.0935, np.nan, 90.8023, np.nan,  42.810, np.nan], # stokeslets
                    [B(cn_h(wavelen / a), ct_h(wavelen / a), L, R, theta) / R ** 2, np.nan,
                     D(cn_h(wavelen / a), ct_h(wavelen / a), L, R, theta) / R ** 3, np.nan,
                     A(cn_h(wavelen / a), ct_h(wavelen / a), L, theta) / R,         np.nan],
                    [B(cn_l(wavelen / a, theta), ct_l(wavelen / a, theta), L, R, theta) / R ** 2, np.nan,
                     D(cn_l(wavelen / a, theta), ct_l(wavelen / a, theta), L, R, theta) / R ** 3, np.nan,
                     A(cn_l(wavelen / a, theta), ct_l(wavelen / a, theta), L, theta) / R,         np.nan],
                    ])

# convert to our normalization
bda_star_mat = bda_mat[4:, :]
bda_star_mat[:, :2] = bda_star_mat[:, :2] * R**2 / (L * R)
bda_star_mat[:, 2:4] = bda_star_mat[:, 2:4] * R**3 / (L * R ** 2)
bda_star_mat[:, 4:] = bda_star_mat[:, 4:] * R / (L)

bda_star_exp_mat = np.array([[np.mean(b_exp), np.std(b_exp), np.mean(d_exp), np.std(d_exp), np.mean(a_exp), np.std(a_exp)],
                               [b_exp[0], b_exp_unc[0], d_exp[0], d_exp_unc[0], a_exp[0], d_exp_unc[0]],
                               [b_exp[1], b_exp_unc[1], d_exp[1], d_exp_unc[1], a_exp[1], d_exp_unc[1]],
                               [b_exp[2], b_exp_unc[2], d_exp[2], d_exp_unc[2], a_exp[2], d_exp_unc[2]]])

bda_star_mat = np.concatenate((bda_star_exp_mat, bda_star_mat))

# print etas
for ii in range(len(viscosities)):
    print(f"viscosity = {viscosities[ii]:.3f}+/-{viscosities_unc[ii]:.3f} mPa*s, {n_expts[ii]:d}, average len = {mean_len_exps[ii]:.2f}um")

# print translational diffusion ratios
for ii in range(len(viscosities)):
    print(f"d parallel / d perp = {d_translation_ratios[ii]:.3f}+/-{d_translation_ratios_unc[ii]:.3f}")

print(f"average flagella length {L:.2f}+/-{std_len:.2f}um")
print(f"average flagella wavelength {wavelen:.3f}+/-{wlen_unc:.3f}um")

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

# print our non-dimensionalization
print("")
print("non-dimensionalized using length and radius:")
print(f"{'':<20}, ", end="")
for ii in range(3):
    print(f"{cols_star[ii]:<20}, ", end="")
print("")

for ii in range(len(rows)):
    print(f"{rows[ii]:<20}, ", end="")
    for jj in range(3):
        print(f"{bda_star_mat[ii, 2*jj]:<10.3f} +-", end="")
        print(f"{bda_star_mat[ii, 2*jj + 1]:<7.3f}, ", end="")
    print("")

# print Rodenborn non-dimensionalization
print("non-dimensionalized using radius:")
print(f"{'':<20}, ", end="")
for ii in range(3):
    print(f"{cols[ii]:<20}, ", end="")
print("")

for ii in range(len(rows)):
    print(f"{rows[ii]:<20}, ", end="")
    for jj in range(3):
        print(f"{bda_mat[ii, 2*jj]:<10.3f} +-", end="")
        print(f"{bda_mat[ii, 2*jj + 1]:<7.3f}, ", end="")
    print("")

# ##################################
# determine sucrose concentrations
# Fig S6
# ##################################
# density of sucrose/water mixture at 20C from Sugar Bett handbook
# g / ml
rho_sucrose_mix = np.array([1000.00, 1003.89, 1007.79, 1011.72, 1015.67, 1017.85, 1023.66,
                            1027.70, 1031.76, 1035.86, 1039.98, 1044.13, 1048.31, 1052.52,
                            1056.77, 1061.04, 1065.34, 1069.68, 1074.04, 1078.44, 1082.87,
                            1087.33, 1091.83, 1096.36, 1100.92, 1105.51, 1110.14, 1114.80,
                            1119.49, 1124.22, 1128.98, 1133.78, 1138.61, 1143.47, 1148.37,
                            1153.31, 1158.28, 1163.29, 1168.33, 1173.41, 1178.53, 1183.68,
                            1188.87, 1194.10, 1199.36, 1204.67, 1210.01, 1215.38, 1220.80,
                            1226.25, 1231.74, 1237.27, 1242.84, 1248.44, 1254.08, 1259.76,
                            1265.48, 1271.23, 1277.03, 1282.86, 1288.73, 1294.64, 1300.59,
                            1306.57, 1312.60, 1318.66, 1324.76, 1330.90, 1337.08, 1343.30,
                            1349.56, 1355.85, 1362.18, 1368.58, 1374.96, 1381.41, 1387.90,
                            1394.42, 1400.98, 1407.58, 1414.21, 1420.88, 1427.59, 1434.34,
                            1441.12, 1447.94, 1454.80, 1461.70, 1468.62, 1475.59, 1482.59,
                            1489.63, 1496.71, 1503.81, 1510.96, 1518.14, 1525.35, 1532.60,
                            1539.88, 1547.19]) * 1e-3
sucrose_by_weight = np.arange(0, 100, 1) / 100

rho_fn = scipy.interpolate.interp1d(sucrose_by_weight, rho_sucrose_mix)

# viscosity in mPa
def visc_model(ww):
    # ww = p[0]

    mw_suc = 342.3 # g/mol
    mw_water = 18.01528 # g/mol

    # molar fraction of sucrose
    xs = (1 / mw_suc) * ww / (ww / mw_suc + (1 - ww) / mw_water)
    # molar volume in L/mol
    vm = (xs * mw_suc + (1 - xs) * mw_water) / rho_fn(ww) * 1e-3
    # viscosity
    eta = 6.31e-3 * np.exp(282 * vm)
    return eta

sucrose_ww_from_viscosities = np.array([scipy.optimize.root_scalar(lambda p: visc_model(p) - v, bracket=[1e-3, 0.99]).root for v in viscosities])
for v, ww in zip(viscosities, sucrose_ww_from_viscosities):
    print(f"For viscosity {v:.2f}mPa*s the inferred sucrose ww concentration is {ww:.3f}")

figh = plt.figure(figsize=(10, 3.), dpi=300)
grid = figh.add_gridspec(nrows=1, ncols=3, wspace=0.35, bottom=0.3)

ax = figh.add_subplot(grid[0, 0])
ax.annotate("A", (-0.05, 1.05), xycoords='axes fraction')
ax.errorbar(initial_sucrose_wv * 100, viscosities, yerr=viscosities_unc,
            marker='o', linestyle="none", color="thistle",
            mew=1, mec="purple", ecolor="purple")

ax.set_xlabel("Initial sucrose w/v [\%]")
ax.set_ylabel("viscosity [mPa $\cdot$ s]")

ax = figh.add_subplot(grid[0, 1])
ax.annotate("B", (-0.05, 1.05), xycoords='axes fraction')
ax.plot(sucrose_by_weight * 100, rho_sucrose_mix,
        marker='.', linestyle="none",
        color="royalblue", ms=3.5,
        mew=0.5, mec="navy")
ax.set_xlabel("w/w sucrose [\%]")
ax.set_ylabel(r"$\rho$ [g/ml]")
ax.set_xlim([0, 100])

ax = figh.add_subplot(grid[0, 2])
ax.annotate("C", (-0.05, 1.05), xycoords='axes fraction')
ax.plot(sucrose_by_weight * 100, visc_model(sucrose_by_weight),
        marker='.', linestyle="none",
        color="royalblue",
        mew=1, mec="navy")
ax.errorbar(sucrose_ww_from_viscosities * 100, viscosities, yerr=viscosities_unc,
            marker='o', linestyle="none",
            color="thistle", ms=3.5,
            mew=1, mec="purple", ecolor="purple")
ax.set_xlabel("w/w sucrose [\%]")
ax.set_ylabel("viscosity [mPa $\cdot$ s]")
ax.set_xlim([0, 40])
ax.set_ylim([0, 6])

if save_plots:
    save_fname = save_dir / f"{tstamp:s}_fig_s6.pdf"
    figh.savefig(save_fname)