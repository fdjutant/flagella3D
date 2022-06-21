"""
Display results of analysis from 2022_06_15_diffusion_analysis.py
"""

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import napari
from pathlib import Path
import tifffile
import numpy as np
import zarr

# prefix = "suc70-h30-03-A"
prefix = "suc40-h15-04-B"

root_dir = Path(r"\\10.206.26.21\flagella_project")
thresholdFolder = root_dir / "threshold-labKit"
intensityFolder = root_dir / "TIF-files"
# proc_folder = root_dir / "2022_06_17_11;01;54_processed_data"
proc_folder = root_dir / "2022_06_20_11;22;38_processed_data"

fname_thresh = list(thresholdFolder.glob(f"{prefix:s}*"))[0]
img_thresh = tifffile.imread(fname_thresh).astype(bool)

fname_int = list(intensityFolder.glob(f"{prefix:s}*"))[0]
img = tifffile.imread(fname_int)

fname_proc = list(proc_folder.glob(f"{prefix:s}*.zarr"))[0]
data = zarr.open(str(fname_proc), "r")

dxyz_um = data.attrs["dxyz_um"]
frame_start, frame_end = data.attrs["frame_range"]
img_thresh_reduced = data.img_thresholded
fitImage = data.helix_fit

# #############################
# visualize
# #############################

# %% View image, threshold, and fit together
viewer = napari.Viewer(ndisplay=3)

viewer.add_image(img[frame_start:frame_end], name="deskewed intensity",
                 contrast_limits=[np.percentile(img[img > 0], 95),
                                  np.percentile(img[img > 0], 99.99)],
                 scale=[dxyz_um, dxyz_um, dxyz_um], blending='additive',
                 multiscale=False, colormap='gray', opacity=1)

viewer.add_image(img_thresh, name="full thresholded image",
                 contrast_limits=[0, 1],
                 scale=[dxyz_um, dxyz_um, dxyz_um], blending='additive',
                 multiscale=False, colormap='green', opacity=0.2, visible=False)

viewer.add_image(img_thresh_reduced, name="reduced thresholded image",
                 contrast_limits=[0, 1],
                 scale=[dxyz_um, dxyz_um, dxyz_um], blending='additive',
                 multiscale=False, colormap='green', opacity=0.2)

viewer.add_image(fitImage[frame_start:frame_end], name="helix fit",
                 contrast_limits=[0, 1],
                 scale=[dxyz_um, dxyz_um, dxyz_um], blending='additive',
                 multiscale=False, colormap='red', opacity=0.2)

viewer.dims.axis_labels = ["time", "z", "y", "x"]
viewer.scale_bar.visible = True
viewer.scale_bar.unit = 'um'
viewer.scale_bar.position = 'top_right'
viewer.axes.visible = True
# napari.run()
