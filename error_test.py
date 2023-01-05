import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
from pyuvs import *
from pathlib import Path
from astropy.io import fits


orbit = 3453
orbit_code = f'orbit' + f'{orbit}'.zfill(5)
block = math.floor(orbit / 100) * 100
orbit_block = 'orbit' + f'{block}'.zfill(5)

dust_files = sorted(Path(f'/home/kyle/iuvs/retrievals/{orbit_block}/data/').glob(f'{orbit_code}-*-dust-radiance.npy'))
ice_files = sorted(Path(f'/home/kyle/iuvs/retrievals/{orbit_block}/data/').glob(f'{orbit_code}-*-ice-radiance.npy'))
chi_squared_files = sorted(Path(f'/home/kyle/iuvs/retrievals/{orbit_block}/data/').glob(f'{orbit_code}-*-error-radiance.npy'))
dust = np.load(dust_files[0]) # np.vstack([np.load(f) for f in dust_files])
ice = np.load(ice_files[0]) #np.vstack([np.load(f) for f in ice_files])
chi_squared = np.load(chi_squared_files[0]) #np.vstack([np.load(f) for f in chi_squared_files])

fig, ax = plt.subplots(1, 3)

ax[0].imshow(dust, vmin=0, vmax=2)
ax[1].imshow(ice, vmin=0, vmax=1)
ax[2].imshow(chi_squared)

plt.savefig(f'/home/kyle/iuvs/retrievals/{orbit_block}/images/{orbit_code}-radiance-test.png', dpi=200)
