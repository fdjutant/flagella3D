import numpy as np

R = 0.25
L = 7.7

# first entry is regularized stokeslets, second is experiments
As = np.array([1.38, 2.2]) * L
Bs = np.array([0.39, 0.53]) * L * R
Ds = np.array([2.93, 3.5]) * L * R**2

R_bacteria = 0.5 # um
Ao = 6*np.pi * R_bacteria
Do = 8*np.pi * R_bacteria**3

# velocity divided by motor angular velocity
vs = Bs * Do / ((As + Ao) * (Ds + Do) - Bs**2)