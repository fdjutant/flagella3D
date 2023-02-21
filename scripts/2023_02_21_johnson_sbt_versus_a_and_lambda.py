"""
Johnson SBT versus parameters
"""

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import datetime
import numpy as np
from pathlib import Path
from scipy.io import loadmat

root_dir = Path(r"\\10.206.26.21\flagella_project\Flagellum_Simulation_GUI\simulation_results\2023_02_21_johnson_sbt_versus_a_lambda")
files = list(root_dir.glob("*.mat"))

save_plots = True

data = []
for f in files:
    data.append(loadmat(f))

# simulation done with R = 1 ... i.e. all parameters normalized by R
R = 0.25
L = data[0]["L"][0, 0] * R

lambdas = data[0]["lambda"].ravel().astype(float)
dlambda = lambdas[1] - lambdas[0]
a = np.array([d["a"][0, 0] for d in data])
da = a[1] - a[0]

# convert to non-dimensionalized form used in paper
As = np.concatenate([d["drag_J"] for d in data], axis=0) * R / (L)
Ds = np.concatenate([d["torque_J"] for d in data], axis=0) * R**3 / (L * R ** 2)
Bs = np.concatenate([d["force_J"] for d in data], axis=0) * R**2 / (L * R)

# experimental values
a_exp = 0.01 / R
lambda_exp = 2.5 / R
A_exp = 2.2
B_exp = 0.53
D_exp = 3.5

imin = np.argmin((As - A_exp)**2 / A_exp**2 + (Bs - B_exp)**2 / B_exp**2 + (Ds - D_exp)**2 / D_exp**2)
ind_min = np.unravel_index(imin, As.shape)

# plot
extent = [lambdas[0] - 0.5 * dlambda, lambdas[-1] + 0.5 * dlambda,
          a[-1] + 0.5 * da, a[0] - 0.5 * da]

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 12})

figh = plt.figure(figsize=(8, 3), dpi=300)
figh.suptitle("Propulsion matrix from Johnson SBT versus filament radius and flagellar wavelength\n"
              f"R={R:.2f}um, L={L:.2f}um")
grid = figh.add_gridspec(nrows=1, ncols=3, wspace=0.75, bottom=0.2, top=0.7)

ax = figh.add_subplot(grid[0, 0])
ax.set_title(r"$\frac{A}{\eta L}$")
im = ax.imshow(As, extent=extent, cmap="bone")
ax.plot(lambda_exp, a_exp, "rx")
ax.set_xlabel(r"$\frac{\lambda}{R}$")
ax.set_ylabel(r"$\frac{a}{R}$")
ax.axis("auto")
plt.colorbar(im)

ax = figh.add_subplot(grid[0, 1])
ax.set_title(r"$\frac{B}{\eta LR}$")
im = ax.imshow(Bs, extent=extent, cmap="bone")
ax.plot(lambda_exp, a_exp, "rx")
ax.set_xlabel(r"$\frac{\lambda}{R}$")
ax.set_ylabel(r"$\frac{a}{R}$")
ax.axis("auto")
plt.colorbar(im)

ax = figh.add_subplot(grid[0, 2])
ax.set_title(r"$\frac{D}{\eta LR^2}$")
im = ax.imshow(Ds, extent=extent, cmap="bone")
ax.plot(lambda_exp, a_exp, "rx")
ax.set_xlabel(r"$\frac{\lambda}{R}$")
ax.set_ylabel(r"$\frac{a}{R}$")
ax.axis("auto")
plt.colorbar(im)

if save_plots:
    save_dir = Path(r"\\10.206.26.21\flagella_project\various-plots")
    tstamp = datetime.datetime.now().strftime('%Y_%m_%d_%H;%M;%S')

    save_fname = save_dir / f"{tstamp:s}_johnson_sbt.pdf"
    figh.savefig(save_fname)
