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

dust_files = sorted(Path(f'/home/kyle/iuvs/retrievals/{orbit_block}/data/').glob(f'{orbit_code}-*-dust-radiance-nm.npy'))
ice_files = sorted(Path(f'/home/kyle/iuvs/retrievals/{orbit_block}/data/').glob(f'{orbit_code}-*-ice-radiance-nm.npy'))
chi_squared_files = sorted(Path(f'/home/kyle/iuvs/retrievals/{orbit_block}/data/').glob(f'{orbit_code}-*-chi_squared-radiance-nm.npy'))
dust = np.vstack([np.load(f) for f in dust_files])
ice = np.vstack([np.load(f) for f in ice_files])
chi_squared = np.vstack([np.load(f) for f in chi_squared_files])

'''fig, ax = plt.subplots(1, 3)

ax[0].imshow(dust, vmin=0, vmax=2)
ax[1].imshow(ice, vmin=0, vmax=1)
ax[2].imshow(chi_squared)

plt.savefig(f'/home/kyle/iuvs/retrievals/{orbit_block}/images/{orbit_code}-radiance.png', dpi=200)
raise SystemExit(9)'''


files = sorted(Path(f'/media/kyle/McDataFace/iuvsdata/production/{orbit_block}').glob(f'*apoapse*{orbit_code}*muv*.gz'))
files = [fits.open(f) for f in files]

latitude = np.vstack([f['pixelgeometry'].data['pixel_corner_lat'][..., 4] for f in files])
longitude = np.vstack([f['pixelgeometry'].data['pixel_corner_lon'][..., 4] for f in files])
altitude = np.vstack([f['pixelgeometry'].data['pixel_corner_mrh_alt'][..., 4] for f in files])
fov = np.concatenate([f['integration'].data['fov_deg'] for f in files])
swath_number = swath_number(fov)


def make_swath_grid(field_of_view: np.ndarray, swath_number: int,
                    n_positions: int, n_integrations: int) \
        -> tuple[np.ndarray, np.ndarray]:
    """Make a swath grid of mirror angles and spatial bins.

    Parameters
    ----------
    field_of_view: np.ndarray
        The instrument's field of view.
    swath_number: int
        The swath number.
    n_positions: int
        The number of positions.
    n_integrations: int
        The number of integrations.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The swath grid.

    """
    slit_angles = np.linspace(angular_slit_width * swath_number,
                              angular_slit_width * (swath_number + 1),
                              num=n_positions+1)
    mean_angle_difference = np.mean(np.diff(field_of_view))
    field_of_view = np.linspace(field_of_view[0] - mean_angle_difference / 2,
                                field_of_view[-1] + mean_angle_difference / 2,
                                num=n_integrations + 1)
    return np.meshgrid(slit_angles, field_of_view)


fig, ax = plt.subplots(3, 1, figsize=(3, 9))

for swath in np.unique(swath_number):
    # Do this no matter if I'm plotting primary or angles
    swath_inds = swath_number == swath
    n_integrations = np.sum(swath_inds)
    x, y = make_swath_grid(fov[swath_inds], swath, 133, n_integrations)
    dax = ax[0].pcolormesh(x, y, dust[swath_inds], linewidth=0, edgecolors='none', rasterized=True)
    iax = ax[1].pcolormesh(x, y, ice[swath_inds], linewidth=0, edgecolors='none', rasterized=True)
    ax[2].pcolormesh(x, y, chi_squared[swath_inds], linewidth=0, edgecolors='none', rasterized=True)

divider = make_axes_locatable(ax[0])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(dax, cax=cax, orientation='vertical')

divider = make_axes_locatable(ax[1])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(iax, cax=cax, orientation='vertical')

ax[0].set_title('Dust optical depth')
ax[0].set_xlim(0, angular_slit_width * (swath_number[-1] + 1))
ax[0].set_ylim(minimum_mirror_angle * 2, maximum_mirror_angle * 2)
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[0].set_facecolor('k')

ax[1].set_title('Ice optical depth')
ax[1].set_xlim(0, angular_slit_width * (swath_number[-1] + 1))
ax[1].set_ylim(minimum_mirror_angle * 2, maximum_mirror_angle * 2)
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[1].set_facecolor('k')

ax[2].set_xlim(0, angular_slit_width * (swath_number[-1] + 1))
ax[2].set_ylim(minimum_mirror_angle * 2, maximum_mirror_angle * 2)
ax[2].set_xticks([])
ax[2].set_yticks([])
ax[2].set_facecolor('k')

plt.savefig(f'/home/kyle/iuvs/retrievals/{orbit_block}/images/{orbit_code}-radiance.png', dpi=200)
