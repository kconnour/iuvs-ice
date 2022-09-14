import numpy as np
import glob
import matplotlib.pyplot as plt


ff = np.load('/home/kyle/repos/PyUVS/pyuvs/anc/flatfields/mid-hi-res-flatfield-update.npy')

#old = np.load('/home/kyle/iuvs/reflectance/orbit03400/reflectance3467-05.npy')
new = np.load('/home/kyle/iuvs/reflectance/orbit03400/reflectance3467-07-nonlinear-solstice.npy') / ff


fig, ax = plt.subplots(1, 2)

ax[0].imshow(np.sum(new[..., 1:4], axis=-1), vmin=0.3, vmax=0.6, cmap='viridis')
ax[1].imshow(np.sum(new[..., -4:-1], axis=-1), vmin=0.1, vmax=0.5, cmap='viridis')

plt.savefig('/home/kyle/iuvs/reflectance-3467-08.png')