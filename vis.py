import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


file: int = 11
dust = np.load(f'/home/kyle/iuvs/retrievals/orbit03400/orbit3464-{file}-dust.npy')
ice = np.load(f'/home/kyle/iuvs/retrievals/orbit03400/orbit3464-{file}-ice.npy')

fig, ax = plt.subplots(1, 1)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
im = ax.imshow(dust, vmin=0, vmax=0.5)
fig.colorbar(im, cax=cax, orientation='vertical')
plt.savefig(f'/home/kyle/iuvs/retrievals/orbit03400/orbit3464-{file}-dust.png')
plt.close(fig)

fig, ax = plt.subplots(1, 1)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
im = ax.imshow(ice, vmin=0, vmax=0.5)
fig.colorbar(im, cax=cax, orientation='vertical')
plt.savefig(f'/home/kyle/iuvs/retrievals/orbit03400/orbit3464-{file}-ice.png')
plt.close(fig)
