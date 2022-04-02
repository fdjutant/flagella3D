import numpy as np
import matplotlib.pyplot as plt
from numpy import fft
import scipy.optimize

x = np.linspace(-5*np.pi, 5*np.pi, 1001)
dx = x[1] - x[0]

# function which should match data
def model_fn(x, p):
    v = p[0] * np.sin(2*np.pi * p[1] * x + p[2]) + p[3]
    return v

def model_jacobian(x, p):
    dp0 = np.sin(2*np.pi * p[1] * x + p[2])
    dp1 = p[0] * (2 * np.pi * x) * np.cos(2*np.pi * p[1] * x + p[2])
    dp2 = p[0] * np.cos(2*np.pi * p[1] * x + p[2])
    dp3 = np.ones(x.shape)
    return np.stack((dp0, dp1, dp2, dp3), axis=1)

true_params = [0.23626, 0.2367, 1.246236, 0]
y = model_fn(x, true_params)
y_noisy = y + np.random.normal(scale=0.05, size=y.shape)

# function which should be minimized ... or rather least_squares will take the sum of the squares to minimize it...
def min_fn(p): return model_fn(x, p) - y_noisy

# need good guess of initial frequency for fit to converge. Get this from FFT of data
y_ft = fft.fftshift(fft.fft(fft.ifftshift(y)))
frqs = fft.fftshift(fft.fftfreq(x.size, dx))
df = frqs[1] - frqs[0]

# guess parameters
peak_index = np.argmax(np.abs(y_ft) * (frqs > 0)) # restrict to positive peak
f_guess = frqs[peak_index]
amp_guess = 2 / x.size * np.abs(y_ft[peak_index])
phi_guess = np.angle(y_ft[peak_index])
bg_guess = np.mean(y_noisy)

# sample fitting without jacobian
init_params = np.array([amp_guess, f_guess, phi_guess, bg_guess])
lower_bounds = [0, 0, -np.inf, -np.inf]
upper_bounds = [np.inf, np.inf, np.inf, np.inf]
results = scipy.optimize.least_squares(min_fn, init_params,
                                       bounds=(lower_bounds, upper_bounds),
                                       jac=lambda p: model_jacobian(x, p))
fit_params = results["x"]

# interpolated fit function
x_interp = np.linspace(x.min(), x.max(), 1001)
y_guess = model_fn(x_interp, init_params)
y_fit = model_fn(x_interp, fit_params)

figh = plt.figure()
figh.suptitle("sample fitting with scipy.optimize")
ax = figh.add_subplot(1, 1, 1)
ax.plot(x, y_noisy, 'rx', label="noisy data")
#ax.plot(x_interp, y_guess, 'k', label="initial guess")
ax.plot(x_interp, y_fit, 'b', label="fit")
ax.set_xlabel("x")
ax.set_ylabel("y")

plt.show()