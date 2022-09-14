import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm


orbit = 3464
file: int = 4
dust = np.load(f'/home/kyle/iuvs/retrievals/orbit03400/data/orbit{orbit}-{file}-dust.npy')
ice = np.load(f'/home/kyle/iuvs/retrievals/orbit03400/data/orbit{orbit}-{file}-ice.npy')
chisq = np.load(f'/home/kyle/iuvs/retrievals/orbit03400/data/orbit{orbit}-{file}-chi_squared.npy')


'''dust = np.vstack((np.load(f'/home/kyle/iuvs/retrievals/orbit03400/data/orbit{orbit}-{7}-dust.npy'),
                  np.load(f'/home/kyle/iuvs/retrievals/orbit03400/data/orbit{orbit}-{8}-dust.npy'),
                  np.load(f'/home/kyle/iuvs/retrievals/orbit03400/data/orbit{orbit}-{9}-dust.npy')))
ice = np.vstack((np.load(f'/home/kyle/iuvs/retrievals/orbit03400/data/orbit{orbit}-{7}-ice.npy'),
                  np.load(f'/home/kyle/iuvs/retrievals/orbit03400/data/orbit{orbit}-{8}-ice.npy'),
                  np.load(f'/home/kyle/iuvs/retrievals/orbit03400/data/orbit{orbit}-{9}-ice.npy')))
chisq = np.vstack((np.load(f'/home/kyle/iuvs/retrievals/orbit03400/data/orbit{orbit}-{7}-chi_squared.npy'),
                  np.load(f'/home/kyle/iuvs/retrievals/orbit03400/data/orbit{orbit}-{8}-chi_squared.npy'),
                  np.load(f'/home/kyle/iuvs/retrievals/orbit03400/data/orbit{orbit}-{9}-chi_squared.npy')))'''
#dust[chisq >= 0.001] = np.nan
#ice[chisq >= 0.001] = np.nan
cmap = cm.get_cmap('viridis').copy()
cmap.set_bad('gray')

fig, ax = plt.subplots(1, 1)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
im = ax.imshow(np.flipud(dust), cmap=cmap, vmin=0, vmax=1)
#ax.set_xticks([])
#ax.set_yticks([])
fig.colorbar(im, cax=cax, orientation='vertical')
plt.savefig(f'/home/kyle/iuvs/retrievals/orbit03400/img/orbit{orbit}-{file}-dust.png')
plt.close(fig)

fig, ax = plt.subplots(1, 1)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
im = ax.imshow(np.flipud(ice), cmap=cmap, vmin=0, vmax=1)
#ax.set_xticks([])
#ax.set_yticks([])
fig.colorbar(im, cax=cax, orientation='vertical')
plt.savefig(f'/home/kyle/iuvs/retrievals/orbit03400/img/orbit{orbit}-{file}-ice.png')
plt.close(fig)

fig, ax = plt.subplots(1, 1)
divider = make_axes_locatable(ax)
cax = divider.append_axes('right', size='5%', pad=0.05)
im = ax.imshow(np.flipud(chisq), cmap=cmap, vmin=0, vmax=0.001)
#ax.set_xticks([])
#ax.set_yticks([])
fig.colorbar(im, cax=cax, orientation='vertical')
plt.savefig(f'/home/kyle/iuvs/retrievals/orbit03400/img/orbit{orbit}-{file}-chisq.png')
plt.close(fig)
