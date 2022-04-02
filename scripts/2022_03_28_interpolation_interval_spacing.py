xx.shape
yy.shape
zz.shape

plt.plot(xx, yy, 'r.')
sort_ind = np.argsort(xx)
x_interp = np.linspace(xx.min(), xx.max(), 1000)
y_interp = np.interp(x_interp, xx[sort_ind], yy[sort_ind])
plt.plot(x_interp, y_interp, 'b')
import numpy.fft as fft
y_ft = fft.fftshift(fft.fft(fft.ifftshift(y_interp)))
dx = xx[sort_ind][1] - xx[sort_ind][0]
yfrqs = fft.fftshift(fft.fftfreq(y_interp.size, dx))
plt.plot(yfrqs, np.abs(y_ft)**2)
ff_allowed = np.logical_and(yfrqs > 0.01, yfrqs < 0.5)
ind = np.argmax(np.abs(y_ft) * ff_allowed)
ind
np.abs(y_ft[ind])
yfrqs[ind]
1 / yfrqs[ind]
z_interp = np.interp(x_interp, xx, zz)
z_ft = fft.fftshift(fft.fft(fft.ifftshift(z_interp)))
plt.plot(yfrqs, np.abs(z_ft)**2)
plt.plot(xx, zz)
z_interp = np.interp(x_interp, xx[sort_ind], zz[sort_ind])
z_ft = fft.fftshift(fft.fft(fft.ifftshift(z_interp)))
plt.plot(x_interp, z_interp, '.')
plt.plot(yfrqs, np.abs(z_ft)**2)