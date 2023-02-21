import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import h5py

# ONI localizations
# root_dir = Path(r"\\10.206.26.21\flagella_project\20220516_TS_100nm\buffer\2023_02_10_15;40;40_diffusion_coeff_using_ONI_localizations")
# root_dir = Path(r"\\10.206.26.21\flagella_project\20220516_TS_100nm\50suc\2023_02_10_17;38;24_diffusion_coeff_using_ONI_localizations")
# root_dir = Path(r"\\10.206.26.21\flagella_project\20220516_TS_100nm\70suc\2023_02_10_17;42;07_diffusion_coeff_using_ONI_localizations")


# new localizations
# root_dir = Path(r"\\10.206.26.21\flagella_project\Tetraspek_100nm_sucrose\20220516_Franky_TS_diffusion_HILO_BufferOnly\2023_02_13_14;36;44_diffusion_coeff")
# root_dir = Path(r"\\10.206.26.21\flagella_project\Tetraspek_100nm_sucrose\20220114_Franky_2x_TS100nm_40suc_MBT\2023_02_14_17;27;33_diffusion_coeff")
root_dir = Path(r"\\10.206.26.21\flagella_project\Tetraspek_100nm_sucrose\20220516_Franky_TS_diffusion_HILO_40sucrose\2023_02_16_11;30;48_diffusion_coeff")
# root_dir = Path(r"\\10.206.26.21\flagella_project\Tetraspek_100nm_sucrose\20220516_Franky_TS_diffusion_HILO_50sucrose\2023_02_13_12;13;40_diffusion_coeff")
# root_dir = Path(r"\\10.206.26.21\flagella_project\Tetraspek_100nm_sucrose\20220516_Franky_TS_diffusion_HILO_70sucrose\2023_02_13_14;36;54_diffusion_coeff")
files = list(root_dir.glob("*hdf5"))

kb = 1.38e-23
T = 273 + 25
R = 99e-3 / 2

ds_msd = []
ds_step = []
nmin_trajs = []
memories = []
search_radii = []

for fname in files:
    with h5py.File(fname, "r") as f:
        ds_msd.append(f.attrs["diffusion_coeff_msd"])
        ds_step.append(f.attrs["diffusion_coeff_steps"])
        nmin_trajs.append(f.attrs["min_trajectory_length"])
        memories.append(f.attrs["memory"])
        search_radii.append(f.attrs["search_radius_um"])

ds_msd = np.array(ds_msd)
ds_step = np.array(ds_step)
nmin_trajs = np.array(nmin_trajs)
memories = np.array(memories)
search_radii = np.array(search_radii)

rs = np.unique(search_radii)
ms = np.unique(memories)
ntrajs = np.unique(nmin_trajs)

# plot results
figh = plt.figure(figsize=(20, 7))
figh.suptitle(f"{str(root_dir.parent.name):s}/{str(root_dir.name):s}")

ax1 = figh.add_subplot(1, 2, 1)
ax1.set_title("$\eta$ from MSD")

ax2 = figh.add_subplot(1, 2, 2)
ax2.set_title("$\eta$ from step size")

for m in ms:
    for nt in ntrajs:
        to_use = np.logical_and(memories == m, nmin_trajs == nt)

        rad = np.unique(search_radii[to_use])

        ds_msd_now = np.zeros(len(rad))
        ds_msd_unc_now = np.zeros(len(rad))
        ds_step_now = np.zeros(len(rad))
        ds_step_unc_now = np.zeros(len(rad))
        for ii, r in enumerate(rad):
            ds_msd_now[ii] = np.mean(ds_msd[to_use][search_radii[to_use] == r])
            ds_msd_unc_now[ii] = np.std(ds_msd[to_use][search_radii[to_use] == r])
            ds_step_now[ii] = np.mean(ds_step[to_use][search_radii[to_use] == r])
            ds_step_unc_now[ii] = np.std(ds_step[to_use][search_radii[to_use] == r])

        ax1.errorbar(rad,
                     kb * T / (6 * np.pi * ds_msd_now * 1e-6**2 * R * 1e-6) * 1e3,
                     yerr=kb * T / (6 * np.pi * ds_msd_now * 1e-6**2 * R * 1e-6) * ds_msd_unc_now / ds_msd_now * 1e3,
                     fmt='.',
                     label=f"memory={m:d}, nmin={nt:d}")
        ax2.errorbar(rad,
                     kb * T / (6 * np.pi * ds_step_now * 1e-6**2 * R * 1e-6) * 1e3,
                     yerr=kb * T / (6 * np.pi * ds_step_now * 1e-6 ** 2 * R * 1e-6) * ds_step_unc_now / ds_step_now * 1e3,
                     fmt='.',
                     label=f"memory={m:d}, nmin={nt:d}")

ax1.legend()
ax2.legend()

# ax1.set_ylim([0, 1.5 * np.max(ds_msd)])
# ax2.set_ylim([0, 1.5 * np.max(ds_msd)])

ax1.set_xlabel("search radius (um)")
ax1.set_ylabel("$\eta$ mPa * s")
ax2.set_xlabel("search radius (um)")
ax2.set_ylabel("$\eta$ mPa * s")

# for buffer, 50, 70
to_use = np.logical_and.reduce((memories == 2, nmin_trajs == 5, search_radii == np.unique(search_radii)[5]))

# for 40
# to_use = np.logical_and.reduce((memories == 0, nmin_trajs == 5, search_radii == np.unique(search_radii)[2]))

d = np.mean(ds_msd[to_use])
dunc = np.std(ds_msd[to_use])
print(f"D = {d:.3f}+/-{dunc:.3f} um^2/s")

eta = kb * T / (6 * np.pi * d * 1e-6**2 * R * 1e-6) * 1e3
eta_unc = eta * dunc / d
print(f"eta = {eta:.3f}+/-{eta_unc:.3f} mPa*s")