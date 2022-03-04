from pathlib import Path
import numpy as np
import dask.array as da
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time
import napari
import os.path
import glob
import pickle
import scipy.signal
from scipy import stats
from skimage.restoration import denoise_tv_chambolle

# import files from this code base
import sys
sys.path.insert(0, 'modules')
import msd
from matmatrix import consistentPCA, hullAnalysis, EuAngfromN


# pixel sizes and exposure rates
px_um = 0.115
camExposure_ms = 2
sweep_um = 15
stepsize_nm = 400
exp3D_ms = 1e-3 * camExposure_ms * (sweep_um * 1e3 / stepsize_nm)

# load data
search_dir = Path(r"\\10.206.26.21\opm2\franky-sample-images")
intensityFiles = list(Path(search_dir).glob("*A.npy"))
thresholdFiles = list(Path(search_dir).glob("*threshold.npy"))

imgs = da.from_npy_stack(intensityFiles[0])
imgs_thresh = da.from_npy_stack(thresholdFiles[0])

# int_raw = np.array(imgs[0])
# int_median_filt = scipy.signal.medfilt(int_raw, kernel_size=(3, 3, 3))
# thresh_median_filt = int_median_filt > 0.8 * np.max(int_median_filt)

# int_tv_denoising = denoise_tv_chambolle(int_raw, weight=0.001)
# int_tv_denoising = int_tv_denoising / np.max(int_tv_denoising)

# generate 3D coordinates for data
nt, nz, ny, nx = imgs.shape
zz, yy, xx = np.meshgrid(px_um * np.arange(nz), px_um * np.arange(ny), px_um * np.arange(nx), indexing="ij")

# analysis
n1s = np.zeros((nt, 3))
n2s = np.zeros((nt, 3))
n3s = np.zeros((nt, 3))
r_coms = np.zeros((nt, 3))
flagella_len = np.zeros(nt)
radial_dist_pt = np.zeros(nt)

tstart = time.perf_counter()
for frame in range(len(imgs_thresh)):
    print("processing frame %d/%d, elapsed time = %0.2fs" % (frame + 1, nt, time.perf_counter() - tstart), end="\r")

    # grab current image
    img_now = np.array(imgs[frame])

    # median filter image
    img_now_med = scipy.signal.medfilt(img_now, kernel_size=(3, 3, 3))

    # threshold image
    img_thresh_med = img_now_med > 0.8 * np.max(img_now_med)

    # find center of mass
    r_coms[frame, 0] = np.sum(xx * img_thresh_med) / np.sum(img_thresh_med)
    r_coms[frame, 1] = np.sum(yy * img_thresh_med) / np.sum(img_thresh_med)
    r_coms[frame, 2] = np.sum(zz * img_thresh_med) / np.sum(img_thresh_med)

    # get coordinates above threshold
    coords = np.vstack((xx[img_thresh_med] - r_coms[frame, 0],
                        yy[img_thresh_med] - r_coms[frame, 1],
                        zz[img_thresh_med] - r_coms[frame, 2])).transpose()

    # ####################################
    # determine axis n1 from PCA and consistency with previous point
    # ####################################
    # pca = PCA(n_components=3)
    pca = PCA(n_components=1)
    pca.fit(coords)
    n1s[frame] = pca.components_[0]

    # choose the sign of current n1 so it is as close as possible to n1 at the previous timestep
    if frame > 0 and np.linalg.norm(n1s[frame] - n1s[frame -1]) > np.linalg.norm(n1s[frame] + n1s[frame - 1]):
        n1s[frame] = -n1s[frame]

    # ####################################
    # determine the flagella length along the vector n1
    # ####################################
    dist_projected_along_n1 = n1s[frame, 0] * coords[:, 0] + n1s[frame, 1] * coords[:, 1] + n1s[frame, 2] * coords[:, 2]
    flagella_len[frame] = np.max(dist_projected_along_n1) - np.min(dist_projected_along_n1)

    # ####################################
    # find the furthest point along the flagella and the positive n1 direction
    # and use this to determine n2
    # ####################################
    ind_pt = np.argmax(dist_projected_along_n1)
    coord_pt = coords[ind_pt]

    # project out n1
    coord_pt_proj = coord_pt - (coord_pt.dot(n1s[frame])) * n1s[frame]

    # check the radial distance of this point from the center
    radial_dist_pt[frame] = np.linalg.norm(coord_pt_proj)

    # generate n2 from this
    n2s[frame] = coord_pt_proj / np.linalg.norm(coord_pt_proj)

    assert n1s[frame].dot(n2s[frame]) < 1e-12

    # generate n3 such that coordinate system is right-handed
    n3s[frame] = np.cross(n1s[frame], n2s[frame])
print("")

n1_angles = np.arccos(np.array([np.dot(n1s[ii], n1s[ii + 1]) for ii in range(nt - 1)]))

# compute displacements and Euler angles
