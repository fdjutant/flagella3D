"""
aggregate data from all datasets and plot diffusion constants
"""
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np
import zarr
import re
from scipy.optimize import least_squares
from pathlib import Path

pts_in_fit = [5, 15, 10]
kb = 1.380649e-23
T = 273 + 25

# data_dir = Path(r"\\10.206.26.21\flagella_project\2022_06_16_13;43;26_processed_data")
data_dir = Path(r"\\10.206.26.21\flagella_project\2022_06_16_17;15;48_processed_data")

patterns = ["suc40*", "suc50*", "suc70*"]
xlims = np.array([0.00166, 0.00295]) * 1e3
xtics = np.array([0.00177, 0.00199, 0.00284]) * 1e3

for n_msd_points_used_in_fit in pts_in_fit:
    figh = plt.figure(figsize=(25, 15))
    figh.suptitle(f"Diffusion and codiffusion coefficients, MSD fits use the first {n_msd_points_used_in_fit:d} points")
    grid = figh.add_gridspec(nrows=6, ncols=6, hspace=0.6, wspace=0.6)
    axs = [[[] for x in range(6)] for y in range(6)]
    for ii in range(6):
        for jj in range(6):
            if ii >= jj:
                axs[ii][jj] = figh.add_subplot(grid[ii, jj])
                if ii == 5:
                    axs[ii][jj].set_xlabel("viscosity (mPa $\cdot$ s)")

                if ii < 3 and jj < 3:
                    axs[ii][jj].set_ylabel("D ($\mu m^2 / s$)")
                elif ii >=3 and jj >= 3:
                    axs[ii][jj].set_ylabel("D (rad$^2 / s$)")
                else:
                    axs[ii][jj].set_ylabel("D (rad $\cdot$ $\mu m / s$)")

                axs[ii][jj].set_title(f"D({ii:d}, {jj:d})")

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

        diff_coeffs = np.zeros((6, 6, len(data_files)))
        diff_coeffs_unc = np.zeros((6, 6, len(data_files)))
        prop_mats = np.zeros((6, 6, len(data_files)))
        for jj in range(len(data_files)):
            data = zarr.open(str(data_files[jj]), "r")
            diff_coeffs[:, :, jj] = data.diffusion_constants[..., n_msd_points_used_in_fit]
            diff_coeffs_unc[:, :, jj] = data.diffusion_constants_unc[..., n_msd_points_used_in_fit]

            prop_mats[:, :, jj] = data.propulsion_matrix[..., n_msd_points_used_in_fit]

            visc = data.attrs["viscosity"] * 1e3
            viscosities[ii] = visc


        for aa in range(6):
            for bb in range(6):
                if aa >= bb:
                    # zero line
                    axs[aa][bb].plot(xlims, [0, 0], 'k')

                    # jitter plot
                    scale = np.min(np.diff(xtics)) / 10
                    x_coord = visc + np.random.uniform(-scale, scale, size=len(diff_coeffs[aa, bb]))

                    axs[aa][bb].errorbar(x_coord, diff_coeffs[aa, bb], yerr=diff_coeffs_unc[aa, bb], fmt='b.')

                    # mean value
                    mean_val = np.mean(diff_coeffs[aa, bb])
                    d_avg[aa, bb, ii] = mean_val
                    d_avg[bb, aa, ii] = mean_val

                    stdm = np.std(diff_coeffs[aa, bb]) / np.sqrt(len(diff_coeffs[aa, bb]))
                    d_stdm[aa, bb, ii] = stdm
                    d_stdm[bb, aa, ii] = stdm

                    axs[aa][bb].errorbar(visc, mean_val, xerr=0.1*visc, yerr=stdm, fmt="r.")

                    axs[aa][bb].set_xlim(xlims)
                    axs[aa][bb].set_xticks(xtics)


        # uncertainty in det(M) is the sum of the adjugate matrix multiplied entrywise by the uncertainty elements
        det_d_unc = np.linalg.det(d_avg[..., ii]) * np.sum(np.linalg.inv(d_avg[..., ii]) * d_stdm[..., ii])

        propulsion_mat_from_avg[..., ii] = kb * T * np.linalg.inv(d_avg[..., ii])
        avg_propulsion_mat = np.mean(prop_mats, axis=-1)



    figh.savefig(data_dir / f"diffusion_constant_summary_msd_pts={n_msd_points_used_in_fit:d}.png")

    # draw tables of diffusion constants
    columns = ["$n_1$", "$n_2$", "$n_3$", "$\phi_1$", "$\phi_2$", "$\phi_3$"]
    rows = columns

    figh = plt.figure(figsize=(25, 15))
    figh.suptitle(f"diffusion constants, MSD points used in fit = {n_msd_points_used_in_fit:d}")
    grid = figh.add_gridspec(nrows=3, ncols=1, hspace=0.4)

    for kk in range(3):
        ax = figh.add_subplot(grid[kk, 0])
        ax.set_title(f"$\eta$ = {viscosities[kk]:.3f} mPa $\cdot$ s")
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
                      rowLabels=rows, colLabels=columns, bbox=[0, 0, 1, 1])
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



        figh.savefig(data_dir / f"diffusion_constant_table_msd_pts={n_msd_points_used_in_fit:d}.png")