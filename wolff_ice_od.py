from astropy.io import fits
import matplotlib.pyplot as plt


hdul = fits.open('/home/kyle/Downloads/cld_2016_189_189.fits.gz')
hdul.info()
raise SystemExit(9)

fig, ax = plt.subplots()
ax.imshow(hdul['tauice'].data, vmin=0, vmax=0.8)
plt.savefig('/home/kyle/iuvs/aaa.png')