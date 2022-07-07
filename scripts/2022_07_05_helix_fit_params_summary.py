"""
aggregate data from all data sets and get average flagella parameters
to generate data used in analysis, see 2022_06_15_diffusion_analysis
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

data_dir = Path(r"\\10.206.26.21\flagella_project\2022_06_20_16;04;38_processed_data")

files = list(data_dir.glob("*.zarr"))
nfiles = len(files)
lengths = np.zeros(nfiles)
rads = np.zeros(nfiles)
pitchs = np.zeros(nfiles)

for ii, f in enumerate(files):
    data = zarr.open(f)
    lengths[ii] = np.mean(data.lengths)
    rads[ii] = np.mean(data.helix_fit_params[:, 0]) * data.attrs["dxyz_um"]
    pitchs[ii] = 2*np.pi / np.mean(data.helix_fit_params[:, 1]) * data.attrs["dxyz_um"]