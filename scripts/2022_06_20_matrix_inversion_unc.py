"""
Estimate uncertainty in coefficients after matrix inversion

Aid to understanding what uncertainty should be for propulsion matrix coefficients when propagated
from diffusion coefficients
"""
import numpy as np

def print_mat(m):
    for ii in range(m.shape[0]):
        for jj in range(m.shape[1]):
            print(f"{m[ii, jj]:>+5.4f}", end="")
            if jj != m.shape[1] - 1:
                print(", ", end="")
            else:
                print("")

def minor_mat(m, r, s):
    m_nw = m[:r, :s]
    m_ne = m[:r, s + 1:]
    m_n = np.concatenate((m_nw, m_ne), axis=1)
    m_sw = m[r + 1:, :s]
    m_se = m[r + 1:, s + 1:]
    m_s = np.concatenate((m_sw, m_se), axis=1)
    minor_mat = np.concatenate((m_n, m_s), axis=0)
    return minor_mat


def minor_det(m, r, s):
    return np.linalg.det(minor_mat(m, r, s))

def get_det_unc(mat, mat_unc):
    det_unc_sqr = 0
    for ii in range(mat.shape[0]):
        for jj in range(mat.shape[1]):
            det_unc_sqr += (mat_unc[ii, jj] * (-1) ** (ii + jj) * minor_det(mat, ii, jj)) ** 2

    return np.sqrt(det_unc_sqr)


mat = np.array([[1,   0,    0, 0.1, 0, 0],
                [0,   0.5,  0, 0,   0, 0],
                [0,   0,  0.5, 0,   0, 0],
                [0.1, 0,    0, 5,   0, 0],
                [0,   0,    0, 0,   1, 0],
                [0,   0,    0, 0,   0, 1]
                ])

a_inv_gt = np.linalg.inv(mat)

# # test inverse matrix construction
# c_mat = np.zeros(mat.shape)
# for ii in range(mat.shape[0]):
#     for jj in range(mat.shape[1]):
#         c_mat[ii, jj] = minor_det(mat, ii, jj) * (-1)**(ii + jj)
# test = np.transpose(c_mat) / np.linalg.det(mat)
# assert np.max(np.abs(test - a_inv_gt)) < 1e-14

# unc = mat * 0.1 + 0.04
unc = mat * 0.01 + 0.02

niterations = 1000000
a_invs = np.zeros((niterations, 6, 6))
a_insts = np.zeros((niterations, 6, 6))
dets_a = np.zeros(niterations)
for ii in range(niterations):
    a_insts[ii] = np.random.normal(mat, unc, size=mat.shape)
    a_invs[ii] = np.linalg.inv(a_insts[ii])

    dets_a[ii] = np.linalg.det(a_insts[ii])

det_unc_num = np.std(dets_a)
a_inst_mean = np.mean(a_insts, axis=0)

# determinant uncertainty from calculation
det_unc_calc = get_det_unc(mat, unc)
print(f"det(A) unc calc = {det_unc_calc:.3f}")
print(f"det(A) unc nume = {det_unc_num:.3f}")

# inverse uncertainty from calculation
cmat_unc_calc = np.zeros((6, 6))
cmat = np.zeros((6, 6))
for ii in range(6):
    for jj in range(6):
        mmat = minor_mat(mat, ii, jj)
        unc_mmat = minor_mat(unc, ii, jj)

        cmat[ii, jj] = (-1)**(ii + jj) * minor_det(mat, ii, jj)

        cmat_unc_calc[ii, jj] = get_det_unc(mmat, unc_mmat)


assert np.max(np.abs(np.transpose(cmat) / np.linalg.det(mat) - a_inv_gt)) < 1e-14

# normally would just add fractional uncertainties, but have to write in slightly different form because some entries are zero
# todo: doesn't seem accurate yet
# maybe problem is I'm using a formula like adj(M) and det(M) are independent ... but they are not...
inv_mat_unc_calc = np.sqrt((cmat_unc_calc.transpose() / np.linalg.det(mat))**2 +
                           (cmat.transpose() / np.linalg.det(mat)**2 * det_unc_calc)**2)


# invert then average
a_inv = np.mean(a_invs, axis=0)
a_inv_unc = np.std(a_invs, axis=0)

# average then invert
a_inv2 = np.linalg.inv(np.mean(a_insts, axis=0))

print("gt")
print_mat(a_inv_gt)
print(f"a inverse average, err={np.sum(np.abs(a_inv_gt - a_inv)):.3f}")
print_mat(a_inv)
print(f"a average inverse, err={np.sum(np.abs(a_inv_gt - a_inv2)):.3f}")
print_mat(a_inv2)
print("a inv unc calculated")
print_mat(inv_mat_unc_calc)
print("a inv unc numerical")
print_mat(a_inv_unc)
print("fractional difference between calcu/numerical unc")
print_mat((inv_mat_unc_calc - a_inv_unc) / (a_inv_unc))
print("fractional uncertainty numerical")
print_mat(a_inv_unc / a_inv)