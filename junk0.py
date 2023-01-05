import numpy as np
import matplotlib.pyplot as plt


'''d = np.load('/home/kyle/iuvs/retrievals/orbit03400/data/orbit03453-03-dust.npy')

fig, ax = plt.subplots()
ax.imshow(d, vmin=0, vmax=2)
ax.set_xticks(np.linspace(0, 130, num=14))
ax.set_yticks(np.linspace(0, 250, num=26))

plt.savefig('/home/kyle/Downloads/diffevo.png')'''

mola = np.load('/home/kyle/repos/iuvs-ice/map/mola-topography.npy')
fig, ax = plt.subplots()
ax.imshow(mola)
plt.savefig('/home/kyle/Downloads/mola.png')