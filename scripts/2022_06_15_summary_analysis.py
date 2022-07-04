"""
aggregate data from all datasets and plot diffusion constants
"""
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 14})
from matplotlib.transforms import Bbox
import numpy as np
import zarr
import re
from scipy.optimize import least_squares
from pathlib import Path

fontsize_big = 24
fontsize_small = 16
# pts_in_fit = [5, 15, 10]
pts_in_fit = [10]
kb = 1.380649e-23
T = 273 + 25

# data_dir = Path(r"\\10.206.26.21\flagella_project\2022_06_16_13;43;26_processed_data")
# data_dir = Path(r"\\10.206.26.21\flagella_project\2022_06_16_17;15;48_processed_data")
# data_dir = Path(r"\\10.206.26.21\flagella_project\2022_06_17_11;01;54_processed_data")
# data_dir = Path(r"\\10.206.26.21\flagella_project\2022_06_20_11;22;38_processed_data")
# data_dir = Path(r"\\10.206.26.21\flagella_project\2022_06_20_13;14;36_processed_data")
data_dir = Path(r"\\10.206.26.21\flagella_project\2022_06_20_16;04;38_processed_data")

patterns = ["suc40*", "suc50*", "suc70*"]
xlims = np.array([0.00166, 0.00295]) * 1e3
xtics = np.array([0.00177, 0.00199, 0.00284]) * 1e3

# functions converting between unit vector and angles
def get_angles(vec):
    phi = np.arctan2(vec[1], vec[0])

    if np.abs(vec[0]) > 1e-12:
        theta = np.arctan2(vec[0] / np.cos(phi), vec[2])
    else:
        theta = np.arctan2(vec[1] / np.sin(phi), vec[2])

    return phi, theta


def get_vector(phi, theta):
    return np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)])

for n_msd_points_used_in_fit in pts_in_fit:
    figh_lab = plt.figure(figsize=(25, 15))
    figh_lab.suptitle(f"Diffusion and codiffusion coefficients in lab frame, MSD fits use the first {n_msd_points_used_in_fit:d} points",
                      fontsize=fontsize_big)
    grid = figh_lab.add_gridspec(nrows=3, ncols=3, hspace=0.6, wspace=0.6)
    axs_lab = [[[] for x in range(3)] for y in range(3)]
    for ii in range(3):
        for jj in range(3):
            if ii >= jj:
                axs_lab[ii][jj] = figh_lab.add_subplot(grid[ii, jj])
                if ii == 2:
                    axs_lab[ii][jj].set_xlabel("viscosity (mPa $\cdot$ s)", fontsize=fontsize_small)

                axs_lab[ii][jj].set_ylabel("D ($\mu m^2 / s$)", fontsize=fontsize_small)

                axs_lab[ii][jj].set_title(f"D({ii:d}, {jj:d})", fontsize=fontsize_small)

    # body-frame diffusion constants
    figh_body = plt.figure(figsize=(25, 15))
    figh_body.suptitle(f"Diffusion and codiffusion coefficients in body frame, MSD fits use the first {n_msd_points_used_in_fit:d} points",
                       fontsize=fontsize_big)
    grid = figh_body.add_gridspec(nrows=6, ncols=6, hspace=0.6, wspace=0.6)
    axs = [[[] for x in range(6)] for y in range(6)]
    for ii in range(6):
        for jj in range(6):
            if ii >= jj:
                axs[ii][jj] = figh_body.add_subplot(grid[ii, jj])
                if ii == 5:
                    axs[ii][jj].set_xlabel("viscosity (mPa $\cdot$ s)", fontsize=fontsize_small)

                if ii < 3 and jj < 3:
                    axs[ii][jj].set_ylabel("D ($\mu m^2 / s$)", fontsize=fontsize_small)
                elif ii >=3 and jj >= 3:
                    axs[ii][jj].set_ylabel("D (rad$^2 / s$)", fontsize=fontsize_small)
                else:
                    axs[ii][jj].set_ylabel("D (rad $\cdot$ $\mu m / s$)", fontsize=fontsize_small)

                axs[ii][jj].set_title(f"D({ii:d}, {jj:d})", fontsize=fontsize_small)

    # load data
    d_avg_lab = np.zeros((3, 3, len(patterns)))
    d_stdm_lab = np.zeros(d_avg_lab.shape)
    d_avg = np.zeros((6, 6, len(patterns)))
    d_stdm = np.zeros((6, 6, len(patterns)))
    propulsion_mat_from_avg = np.zeros(d_avg.shape)
    propulsion_mat_from_avg_unc = np.zeros(d_avg.shape)
    avg_propulsion_mat = np.zeros(d_avg.shape)
    viscosities = np.zeros((len(patterns)))
    for ii, p in enumerate(patterns):
        data_files = list(data_dir.glob(p + ".zarr"))

        if data_files == []:
            continue

        thetas = []
        phis = []
        n_time_lag_max_moments = 180
        diff_coeffs_lab = np.zeros((3, 3, len(data_files)))
        diff_coeffs_unc_lab = np.zeros(diff_coeffs_lab.shape)
        non_gauss_coeff_lab = np.zeros((3, n_time_lag_max_moments, len(data_files)))

        diff_coeffs_body = np.zeros((6, 6, len(data_files)))
        diff_coeffs_unc_body = np.zeros((6, 6, len(data_files)))
        non_gauss_coeff_body = np.zeros((6, n_time_lag_max_moments, len(data_files)))
        prop_mats = np.zeros((6, 6, len(data_files)))

        for jj in range(len(data_files)):
            data = zarr.open(str(data_files[jj]), "r")

            diff_coeffs_lab[:, :, jj] = data.diffusion_constants_lab[:, :, n_msd_points_used_in_fit]
            diff_coeffs_unc_lab[:, :, jj] = data.diffusion_constants_unc_lab[:, :, n_msd_points_used_in_fit]

            diff_coeffs_body[:, :, jj] = data.diffusion_constants_body[..., n_msd_points_used_in_fit]
            diff_coeffs_unc_body[:, :, jj] = data.diffusion_constants_unc_body[..., n_msd_points_used_in_fit]

            # compute non gaussian coefficients
            for dd in range(3):
                non_gauss_coeff_lab[dd, :, jj] = np.array(data.fourth_moments_lab[dd, :n_time_lag_max_moments]) / \
                                                 (3 * np.array(data.msds_lab[dd, dd, :n_time_lag_max_moments]) ** 2) - 1

            for dd in range(6):
                non_gauss_coeff_body[dd, :, jj] = np.array(data.fourth_moments_body[dd, :n_time_lag_max_moments]) / \
                                                  (3 * np.array(data.msds_body[dd, dd, :n_time_lag_max_moments]) ** 2) - 1

            prop_mats[:, :, jj] = data.propulsion_matrix[..., n_msd_points_used_in_fit]

            visc = data.attrs["viscosity"] * 1e3
            viscosities[ii] = visc

            n1s = np.array(data.n1)

            phis_now = np.zeros(len(n1s))
            thetas_now = np.zeros(len(n1s))
            for vv in range(len(n1s)):
                phis_now[vv], thetas_now[vv] = get_angles(n1s[vv])

            thetas.append(thetas_now)
            phis.append(phis_now)
        phis = np.concatenate(phis, axis=0)
        thetas = np.concatenate(thetas, axis=0)

        # plot angular distribution
        figh_angles = plt.figure(figsize=(25, 15))
        ax = figh_angles.add_subplot(1, 1, 1)
        ax.plot(phis, thetas, c=np.array([0, 0, 1, 0.5]), marker=".", markersize=15, linestyle="")
        ax.set_xlabel("$\phi$ (rad)", fontsize=fontsize_small)
        ax.set_ylabel(r"$\theta$ (rad)", fontsize=fontsize_small)
        ax.set_title(f"Angular distribution, pattern = {p:s}")
        figh_angles.savefig(data_dir / f"angular_distribution_pattern={p.replace('*', ''):s}.png")

        # plot lab frame non-gaussian parameters
        figh_nongauss = plt.figure(figsize=(25, 5))
        figh_nongauss.suptitle(f"Non-Gaussian parameters, lab frame, pattern = {p:s}")

        lag_time = np.arange(1, non_gauss_coeff_lab.shape[1] + 1) * data.attrs["volumeteric_exposure_time_ms"] * 1e-3
        for aa in range(3):
            ax = figh_nongauss.add_subplot(1, 3, aa + 1)
            ax.errorbar(lag_time, np.mean(non_gauss_coeff_lab[aa], axis=-1),
                        yerr=np.std(non_gauss_coeff_lab[aa], axis=-1) / np.sqrt(non_gauss_coeff_lab.shape[1]),
                        c=np.array([0, 0, 1, 0.5]), marker=".", markersize=10)
            ax.set_title(f"$G_{{ {aa:d} }}(\delta t)$")
            ax.set_xlabel("lag time (ms)")
        figh_nongauss.savefig(data_dir / f"nongauss_parameter_lab_frame_pattern={p.replace('*', ''):s}.png")


        # plot body-frame frame non-gaussian parameters
        figh_nongauss_body = plt.figure(figsize=(25, 5))
        figh_nongauss_body.suptitle(f"Non-Gaussian parameters, body frame, pattern = {p:s}")

        lag_time = np.arange(1, non_gauss_coeff_body.shape[1] + 1) * data.attrs["volumeteric_exposure_time_ms"] * 1e-3
        for aa in range(6):
            ax = figh_nongauss_body.add_subplot(1, 6, aa + 1)
            ax.errorbar(lag_time, np.mean(non_gauss_coeff_body[aa], axis=-1),
                    yerr=np.std(non_gauss_coeff_body[aa], axis=-1) / np.sqrt(non_gauss_coeff_body.shape[1]),
                    c=np.array([0, 0, 1, 0.5]), marker=".", markersize=10)
            ax.set_title(f"$G_{{ {aa:d} }}(\delta t)$")
            ax.set_xlabel("lag time (ms)")
        figh_nongauss_body.savefig(data_dir / f"nongauss_parameter_body_frame_pattern={p.replace('*', ''):s}.png")

        # plot body frame diffusion constants
        for aa in range(6):
            for bb in range(6):
                if aa >= bb:
                    # zero line
                    axs[aa][bb].plot(xlims, [0, 0], c=np.array([0, 0, 0, 0.5]), linewidth=5)

                    # jitter plot
                    scale = np.min(np.diff(xtics)) / 10
                    x_coord = visc + np.random.uniform(-scale, scale, size=len(diff_coeffs_body[aa, bb]))

                    axs[aa][bb].errorbar(x_coord, diff_coeffs_body[aa, bb], yerr=diff_coeffs_unc_body[aa, bb],
                                         c=np.array([0, 0, 1, 0.25]), marker=".", markersize=10, linestyle="")

                    # mean value
                    mean_val = np.mean(diff_coeffs_body[aa, bb])
                    d_avg[aa, bb, ii] = mean_val
                    d_avg[bb, aa, ii] = mean_val

                    stdm = np.std(diff_coeffs_body[aa, bb]) / np.sqrt(len(diff_coeffs_body[aa, bb]))
                    d_stdm[aa, bb, ii] = stdm
                    d_stdm[bb, aa, ii] = stdm

                    axs[aa][bb].errorbar(visc, mean_val, xerr=0.*visc, yerr=stdm, c=np.array([1, 0, 0, 0.5]),
                                         marker=".", markersize=15, linestyle="")

                    axs[aa][bb].set_xlim(xlims)
                    axs[aa][bb].set_xticks(xtics)

        # plot lab frame diffusion constants
        for aa in range(3):
            for bb in range(3):
                if aa >= bb:
                    axs_lab[aa][bb].plot(xlims, [0, 0], c=np.array([0, 0, 0, 0.5]), linewidth=5)

                    # jitter plot
                    scale = np.min(np.diff(xtics)) / 10
                    x_coord = visc + np.random.uniform(-scale, scale, size=len(diff_coeffs_lab[aa, bb]))

                    axs_lab[aa][bb].errorbar(x_coord, diff_coeffs_lab[aa, bb], yerr=diff_coeffs_unc_lab[aa, bb],
                                             c=np.array([0, 0, 1, 0.25]), marker=".", markersize=10, linestyle="")

                    if aa == bb:
                        d_space_mean = np.mean(np.diag(d_avg[:3, :3, ii]))
                        d_space_mean_unc = np.mean(np.diag(d_stdm[:3, :3, ii]))
                        axs_lab[aa][bb].errorbar(visc, d_space_mean, yerr=d_space_mean_unc,
                                                 c=np.array([0.5, 0, 1, 0.5]), marker=".", markersize=15, linestyle="")

                    d_avg_lab[aa, bb, ii] = np.mean(diff_coeffs_lab[aa, bb])
                    d_avg_lab[bb, aa, ii] = d_avg_lab[aa, bb, ii]

                    d_stdm_lab[aa, bb, ii] = np.std(diff_coeffs_lab[aa, bb]) / np.sqrt(len(diff_coeffs_lab[aa, bb]))
                    d_stdm_lab[bb, aa, ii] = d_stdm_lab[aa, bb, ii]

                    axs_lab[aa][bb].errorbar(visc, d_avg_lab[aa, bb, ii], xerr=0. * visc, yerr=stdm,
                                             c=np.array([1, 0, 0, 0.5]), marker=".", markersize=15, linestyle="")

                    axs_lab[aa][bb].set_xlim(xlims)
                    axs_lab[aa][bb].set_xticks(xtics)

        # uncertainty in det(M) is the sum of the adjugate matrix multiplied entrywise by the uncertainty elements
        # det_d_unc = np.linalg.det(d_avg[..., ii]) * np.sum(np.linalg.inv(d_avg[..., ii]) * d_stdm[..., ii])

        nsamples = 1000
        d_inv_test = np.zeros((6, 6, nsamples))
        for tt in range(nsamples):
            d_inv_test[..., tt] = np.linalg.inv(d_avg[..., ii] + np.random.normal(0, scale=d_stdm[..., ii]))
        dinv_unc = np.std(d_inv_test, axis=-1)

        propulsion_mat_from_avg[..., ii] = kb * T * np.linalg.inv(d_avg[..., ii])
        propulsion_mat_from_avg_unc[..., ii] = kb * T * dinv_unc
        avg_propulsion_mat = np.mean(prop_mats, axis=-1)



    figh_body.savefig(data_dir / f"diffusion_constant_body_frame_summary_msd_pts={n_msd_points_used_in_fit:d}.png")
    figh_lab.savefig(data_dir / f"diffusion_constant_lab_frame_summary_msd_pts={n_msd_points_used_in_fit:d}.png")

    # draw tables of diffusion constants
    columns = ["$n_1$", "$n_2$", "$n_3$", "$\phi_1$", "$\phi_2$", "$\phi_3$"]
    rows = columns

    figh_body = plt.figure(figsize=(25, 15))
    figh_body.suptitle(f"diffusion constants, MSD points used in fit = {n_msd_points_used_in_fit:d}", fontsize=fontsize_big)
    grid = figh_body.add_gridspec(nrows=3, ncols=1, hspace=0.4)

    for kk in range(3):
        ax = figh_body.add_subplot(grid[kk, 0])
        ax.set_title(f"$\eta$ = {viscosities[kk]:.3f} mPa $\cdot$ s", fontsize=fontsize_small)
        ax.set_axis_off()

        text = np.zeros((6, 6)).tolist()
        for ii in range(6):
            for jj in range(6):
                if ii >= jj:
                    text[ii][jj] = str(np.round(d_avg[ii, jj, kk], 3))
                else:
                    text[ii][jj] = ""

        highlight_color = np.array([175/255, 238/255, 238/255, 1])

        cell_colors = np.zeros((6, 6)).tolist()
        for ii in range(6):
            for jj in range(6):
                if ii == jj:
                    cell_colors[ii][jj] = highlight_color
                else:
                    cell_colors[ii][jj] = np.array([0, 0, 0, 0])
        cell_colors[3][0] = highlight_color

        tbl = ax.table(cellText=text, cellColours=cell_colors,
                      rowLabels=rows, colLabels=columns, bbox=[0, 0, 1, 1],
                      fontsize=fontsize_small) #todo: fontsize label doesn't seem to do anything...
                      #loc="center")

        # cells = tbl.get_celld()
        # for ii in range(1, 7):
        #     for jj in range(6):
        #         # set colors
        #         # if ii == jj:
        #         #     cells[(ii, jj)].set_facecolor(highlight_color)
        #
        #         # set boundaries
        #         if (ii == 3 or ii == 6) and (jj == 2 or jj == 5):
        #             cells[(ii, jj)].visible_edges = "BR"
        #         elif ii == 6 and (jj != 2 and jj != 5):
        #             cells[(ii, jj)].visible_edges = "B"
        #         elif ii == 3 and jj != 2:
        #             cells[(ii, jj)].visible_edges = "B"
        #         elif (jj == 2 and ii != 3) or jj == 5:
        #             cells[(ii, jj)].visible_edges = "R"
        #         else:
        #             cells[(ii, jj)].visible_edges = ""

    figh_body.savefig(data_dir / f"diffusion_constant_table_msd_pts={n_msd_points_used_in_fit:d}.png")

    # propulsion matrix
    figh_body = plt.figure(figsize=(25, 15))
    figh_body.suptitle(f"Propulsion matrix, MSD points used in fit = {n_msd_points_used_in_fit:d}",
                       fontsize=fontsize_big)
    grid = figh_body.add_gridspec(nrows=3, ncols=1, hspace=0.4)

    for kk in range(3):
        ax = figh_body.add_subplot(grid[kk, 0])
        ax.set_title(f"$\eta$ = {viscosities[kk]:.3f} mPa $\cdot$ s", fontsize=fontsize_small)
        ax.set_axis_off()

        text = np.zeros((6, 6)).tolist()
        for ii in range(6):
            for jj in range(6):
                if ii >= jj:
                    text[ii][jj] = str(np.round(propulsion_mat_from_avg[ii, jj, kk] / (kb * T), 3))
                else:
                    text[ii][jj] = ""

        highlight_color = np.array([175 / 255, 238 / 255, 238 / 255, 1])

        cell_colors = np.zeros((6, 6)).tolist()
        for ii in range(6):
            for jj in range(6):
                err = np.abs(propulsion_mat_from_avg_unc[ii, jj, kk] / propulsion_mat_from_avg[ii, jj, kk])
                if err > 1:
                    err = 1
                color = np.array([highlight_color[0], highlight_color[1], highlight_color[2], 1 - err])

                if ii >= jj:
                    cell_colors[ii][jj] = color
                else:
                    cell_colors[ii][jj] = np.array([0, 0, 0, 0])


        tbl = ax.table(cellText=text, cellColours=cell_colors,
                       rowLabels=rows, colLabels=columns, bbox=[0, 0, 1, 1],
                       fontsize=fontsize_small)  # todo: fontsize label doesn't seem to do anything...
        # loc="center")

        figh_body.savefig(data_dir / f"propulsion_matrix_table_msd_pts={n_msd_points_used_in_fit:d}.png")