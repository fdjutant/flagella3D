"""
Use RFT expression given in Rodenborn et. al https://doi.org/10.1073/pnas.1219831110
For references on the helix parameters, see Turner et al https://doi.org/10.1128/JB.182.10.2793-2801.2000 and
Namba et al https://doi.org/10.1038/342648a0

"""
import numpy as np

# helix parameter estimates for this experiment
l = 7.7 # length in um
rh = 0.25 # helical radius
dh = 2 * rh
rf = 0.01 # filament radius
pitch = 2. # pitch in um
theta = np.arctan(2*np.pi * rh / pitch) # pitch angle

# compare with Rodenborn matlab code
# rh = 1 # helical radius
# l = 3*rh # length in um
# rf = 0.063 * rh # filament radius
# pitch = 2.42 * rh # pitch in um
# theta = np.arctan(2*np.pi * rh / pitch) # pitch angle

# define propulsion matrix
# F = A*v + B * omega
# T = B*V + D * omega

# as functions of cn and ct, the RFT drag coefficients
def A(cn, ct): return (cn * np.sin(theta)**2 + ct * np.cos(theta)**2) * l / np.cos(theta)
def D(cn, ct): return rh**2 * (cn * np.cos(theta)**2 + ct * np.sin(theta)**2) * l / np.cos(theta)
def B(cn, ct): return rh * (cn - ct) * np.sin(theta) * np.cos(theta) * l / np.cos(theta)

# Hancocks drag coefficients (divided by viscosity)
ct_h = 2*np.pi / (np.log(2*pitch / rf) - 0.5)
cn_h = 4*np.pi / (np.log(2*pitch / rf) + 0.5)

# Lighthill's drag coefficients
ct_l = 2*np.pi / (np.log(0.18*pitch / rf / np.cos(theta)))
cn_l = 4*np.pi / (np.log(0.18*pitch / rf / np.cos(theta)) + 0.5)

# print results
print(f"l={l:0.3f}um")
print(f"helical radius={rh:0.3f}um")
print(f"filament radius={rf:0.3f}um")
print(f"helix pitch={pitch:0.3f}um")
print(f"helix pitch angle={theta * 180/np.pi:0.3f}deg")

print("non-dimensionalized values using both dh and L")
print("Hancock RFT:")
print(f"A*={A(cn_h, ct_h) / l:.3f}")
print(f"D*={D(cn_h, ct_h) / (l * dh**2):.3f}")
print(f"B*={B(cn_h, ct_h) / (l * dh):.3f}")
print(f"eff={B(cn_h, ct_h)**2 / (4 * A(cn_h, ct_h) * D(cn_h, ct_h)):.3f}")


print("Lighthill RFT:")
print(f"A*={A(cn_l, ct_l) / l:.3f}")
print(f"D*={D(cn_l, ct_l) / (l * dh**2):.3f}")
print(f"B*={B(cn_l, ct_l) / (l * dh):.3f}")
print(f"eff={B(cn_l, ct_l)**2 / (4 * A(cn_l, ct_l) * D(cn_l, ct_l)):.3f}")

print("non-dimensionalized values using Rodenborn approach")
print("Hancock RFT:")
print(f"A rb={A(cn_h, ct_h) / rh:.3f}")
print(f"D rb={D(cn_h, ct_h) / rh**3:.3f}")
print(f"B rb={B(cn_h, ct_h) / rh**2:.3f}")

print("Lighthill RFT:")
print(f"A rb={A(cn_l, ct_l) / rh:.3f}")
print(f"D rb={D(cn_l, ct_l) / rh**3:.3f}")
print(f"B rb={B(cn_l, ct_l) / rh**2:.3f}")

