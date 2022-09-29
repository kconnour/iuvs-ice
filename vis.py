import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.cm as cm
from pyuvs import *
from pathlib import Path
from astropy.io import fits
from pyuvs.spice import *
from netCDF4 import Dataset
import mer

# Plot parameters
fig, ax = plt.subplots(3, 4, figsize=(12, 9))
dustvmax = 2
icevmax = 1
errorvmax = 0.1
latmin = -45
latmax = 45
lonmin = 180
lonmax = 270
dustcmap = 'cividis'
icecmap = 'viridis'
errorcmap = 'magma'

#######################
### Add in the IUVS data in QL form
#######################
orbit = 3453
orbit_code = f'orbit' + f'{orbit}'.zfill(5)
block = math.floor(orbit / 100) * 100
orbit_block = 'orbit' + f'{block}'.zfill(5)

dust_files = sorted(Path(f'/home/kyle/iuvs/retrievals/{orbit_block}/data/').glob(f'{orbit_code}-*-dust.npy'))
ice_files = sorted(Path(f'/home/kyle/iuvs/retrievals/{orbit_block}/data/').glob(f'{orbit_code}-*-ice.npy'))
error_files = sorted(Path(f'/home/kyle/iuvs/retrievals/{orbit_block}/data/').glob(f'{orbit_code}-*-error.npy'))
dust = np.vstack([np.load(f) for f in dust_files])
ice = np.vstack([np.load(f) for f in ice_files])
error = np.vstack([np.load(f) for f in error_files])


files = sorted(Path(f'/media/kyle/McDataFace/iuvsdata/production/{orbit_block}').glob(f'*apoapse*{orbit_code}*muv*.gz'))
files = [fits.open(f) for f in files]

lat = np.vstack([f['pixelgeometry'].data['pixel_corner_lat'] for f in files])
lon = np.vstack([f['pixelgeometry'].data['pixel_corner_lon'] for f in files])
lon = np.where(lon < 0, lon + 360, lon)
alt = np.vstack([f['pixelgeometry'].data['pixel_corner_mrh_alt'][..., 4] for f in files])
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


for swath in np.unique(swath_number):
    # Do this no matter if I'm plotting primary or angles
    swath_inds = swath_number == swath
    n_integrations = np.sum(swath_inds)
    x, y = make_swath_grid(fov[swath_inds], swath, 133, n_integrations)
    dax = ax[0, 0].pcolormesh(x, y, dust[swath_inds], linewidth=0, edgecolors='none', rasterized=True, vmin=0, vmax=dustvmax, cmap='cividis')
    iax = ax[1, 0].pcolormesh(x, y, ice[swath_inds], linewidth=0, edgecolors='none', rasterized=True, vmin=0, vmax=icevmax, cmap='viridis')
    eax = ax[2, 0].pcolormesh(x, y, error[swath_inds], linewidth=0, edgecolors='none', rasterized=True, vmin=0, vmax=errorvmax, cmap='magma')

ax[0, 0].set_title('Dust optical depth')
ax[0, 0].set_xlim(0, angular_slit_width * (swath_number[-1] + 1))
ax[0, 0].set_ylim(minimum_mirror_angle * 2, maximum_mirror_angle * 2)
ax[0, 0].set_xticks([])
ax[0, 0].set_yticks([])
ax[0, 0].set_facecolor('gray')

ax[1, 0].set_title('Ice optical depth')
ax[1, 0].set_xlim(0, angular_slit_width * (swath_number[-1] + 1))
ax[1, 0].set_ylim(minimum_mirror_angle * 2, maximum_mirror_angle * 2)
ax[1, 0].set_xticks([])
ax[1, 0].set_yticks([])
ax[1, 0].set_facecolor('gray')

ax[2, 0].set_title('Error')
ax[2, 0].set_xlim(0, angular_slit_width * (swath_number[-1] + 1))
ax[2, 0].set_ylim(minimum_mirror_angle * 2, maximum_mirror_angle * 2)
ax[2, 0].set_xticks([])
ax[2, 0].set_yticks([])
ax[2, 0].set_facecolor('gray')

#######################
### Add in the IUVS data in cylindrical map form
#######################

def latlon_meshgrid(latitude, longitude, altitude):
    # make meshgrids to hold latitude and longitude grids for pcolormesh display
    X = np.zeros((latitude.shape[0] + 1, latitude.shape[1] + 1))
    Y = np.zeros((longitude.shape[0] + 1, longitude.shape[1] + 1))
    mask = np.ones((latitude.shape[0], latitude.shape[1]))

    # loop through pixel geometry arrays
    for i in range(int(latitude.shape[0])):
        for j in range(int(latitude.shape[1])):

            # there are some pixels where some of the pixel corner longitudes are undefined
            # if we encounter one of those, set the data value to missing so it isn't displayed
            # with pcolormesh
            if np.size(np.where(np.isfinite(longitude[i, j]))) != 5:
                mask[i, j] = np.nan

            # also mask out non-disk pixels
            if altitude[i, j] != 0:
                mask[i, j] = np.nan

            # place the longitude and latitude values in the meshgrids
            X[i, j] = longitude[i, j, 1]
            X[i + 1, j] = longitude[i, j, 0]
            X[i, j + 1] = longitude[i, j, 3]
            X[i + 1, j + 1] = longitude[i, j, 2]
            Y[i, j] = latitude[i, j, 1]
            Y[i + 1, j] = latitude[i, j, 0]
            Y[i, j + 1] = latitude[i, j, 3]
            Y[i + 1, j + 1] = latitude[i, j, 2]

    # set any of the NaN values to zero (otherwise pcolormesh will break even if it isn't displaying the pixel).
    X[np.where(~np.isfinite(X))] = 0
    Y[np.where(~np.isfinite(Y))] = 0

    # set to domain [-180,180)
    #X[np.where(X > 180)] -= 360

    # return the coordinate arrays and the mask
    return X, Y


for swath in np.unique(swath_number):
    x, y = latlon_meshgrid(lat[swath==swath_number], lon[swath==swath_number], alt[swath==swath_number])
    cdax = ax[0, 1].pcolormesh(x, y, dust[swath==swath_number], vmin=0, vmax=dustvmax, cmap='cividis')
    ciax = ax[1, 1].pcolormesh(x, y, ice[swath==swath_number], vmin=0, vmax=icevmax, cmap='viridis')
    ceax = ax[2, 1].pcolormesh(x, y, error[swath == swath_number], vmin=0, vmax=errorvmax, cmap='magma')

'''divider = make_axes_locatable(ax[0, 1])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(cdax, cax=cax, orientation='vertical')

divider = make_axes_locatable(ax[1, 1])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(ciax, cax=cax, orientation='vertical')

divider = make_axes_locatable(ax[2, 1])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(ceax, cax=cax, orientation='vertical')'''

divider = make_axes_locatable(ax[2, 1])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(ceax, cax=cax, orientation='vertical')

ax[0, 1].set_title('IUVS Dust')
ax[0, 1].set_xlim(lonmin, lonmax)
ax[0, 1].set_ylim(latmin, latmax)
ax[0, 1].set_facecolor('gray')

ax[1, 1].set_title('IUVS Ice')
ax[1, 1].set_xlim(lonmin, lonmax)
ax[1, 1].set_ylim(latmin, latmax)
ax[1, 1].set_facecolor('gray')

ax[2, 1].set_title('IUVS Error')
ax[2, 1].set_xlim(lonmin, lonmax)
ax[2, 1].set_ylim(latmin, latmax)
ax[2, 1].set_facecolor('gray')

#######################
### Add in the MARCI data
#######################

# Compute SPICE info
spice_path = Path('/media/kyle/McDataFace/spice')
s = Spice(spice_path)
s.load_spice()
orbits, all_et = s.find_all_maven_apsis_et('apoapse', endtime=datetime(2022, 5, 29))

print('found apsis info')
et = all_et[orbits == orbit][0]
pf = PositionFinder(et)
dt = pf.get_datetime()
sol = mer.EarthDateTime(dt.year, dt.month, dt.day).to_sol()

# Get the relevant MARCI file
doy = f'{dt.timetuple().tm_yday}'.zfill(3)
marci = fits.open(f'/home/kyle/iuvs/marci/cld_{dt.year}_{doy}_{doy}.fits.gz')
marci_ice = marci['tauice'].data
marci_ice = np.roll(marci_ice, 1440, axis=1)   # shape: (1440, 2880)
marci_lat = np.broadcast_to(np.linspace(-90, 90, num=1441), (2881, 1441))
marci_lon = np.broadcast_to(np.linspace(0, 360, num=2881), (1441, 2881)).T

iceax = ax[1, 2].pcolormesh(marci_lon, marci_lat, marci_ice.T, vmin=0, vmax=icevmax, cmap=icecmap)

ax[1, 2].set_title('MARCI Ice')
ax[1, 2].set_xlim(lonmin, lonmax)
ax[1, 2].set_ylim(latmin, latmax)
ax[1, 2].set_facecolor('gray')

#######################
### Add in the dust/ice climatology
#######################
yearly_gcm = Dataset('/media/kyle/McDataFace/ames/sim1/c48_big.atmos_average_plev-001.nc')

gcm_dust = yearly_gcm['taudust_VIS'][:]
gcm_ice = yearly_gcm['taucloud_VIS'][:]
gcm_lat = np.broadcast_to(np.linspace(-90, 90, num=91), (181, 91))
gcm_lon = np.broadcast_to(np.linspace(0, 360, num=181), (91, 181)).T
gcm_dust_ax = ax[0, 3].pcolormesh(gcm_lon, gcm_lat, gcm_dust[int(sol/668*140), :, :].T, vmin=0, vmax=dustvmax, cmap=dustcmap)
gcm_ice_ax = ax[1, 3].pcolormesh(gcm_lon, gcm_lat, gcm_ice[int(sol/668*140), :, :].T, vmin=0, vmax=icevmax, cmap=icecmap)

divider = make_axes_locatable(ax[0, 3])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(gcm_dust_ax, cax=cax, orientation='vertical')

divider = make_axes_locatable(ax[1, 3])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(gcm_ice_ax, cax=cax, orientation='vertical')

ax[0, 3].set_title('Ames GCM Dust')
ax[0, 3].set_xlim(lonmin, lonmax)
ax[0, 3].set_ylim(latmin, latmax)

ax[1, 3].set_title('Ames GCM Ice')
ax[1, 3].set_xlim(lonmin, lonmax)
ax[1, 3].set_ylim(latmin, latmax)

# Remove unused plot ticks
ax[0, 2].set_xticks([])
ax[0, 2].set_yticks([])

ax[2, 2].set_xticks([])
ax[2, 2].set_yticks([])

ax[2, 3].set_xticks([])
ax[2, 3].set_yticks([])

plt.savefig(f'/home/kyle/iuvs/retrievals/{orbit_block}/images/{orbit_code}.png', dpi=200)
