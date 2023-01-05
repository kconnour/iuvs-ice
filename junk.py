
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import math
from pathlib import Path


#hdul = fits.open('/home/kyle/Downloads/ieg0125t_s1D_alts.fits')
#topo = np.roll(np.flipud(hdul['primary'].data), 1440, axis=0)   # lat is 90 to -90, lon is now 0 to 360

#np.save('/home/kyle/repos/iuvs-ice/map/mola-topography.npy', topo)
orbit: int = 3453
orbit_code = f'orbit' + f'{orbit}'.zfill(5)
block = math.floor(orbit / 100) * 100
orbit_block = 'orbit' + f'{block}'.zfill(5)
reflectance_files = sorted(Path(f'/home/kyle/iuvs/radiance/{orbit_block}').glob(f'radiance-{orbit_code}*.npy'))

fig, ax = plt.subplots()
ax.imshow(np.load(reflectance_files[3])[..., -1], norm='log', vmin=0.001, vmax=0.1)
plt.savefig('/home/kyle/Downloads/reflect.png')
raise SystemExit(9)

#######################
### Add in the IUVS data in QL form
#######################
orbit = 3453
orbit_code = f'orbit' + f'{orbit}'.zfill(5)
block = math.floor(orbit / 100) * 100
orbit_block = 'orbit' + f'{block}'.zfill(5)

dust_files = sorted(Path(f'/home/kyle/iuvs/retrievals/{orbit_block}/data/').glob(f'{orbit_code}-*-dust-pscale.npy'))
dust = np.vstack([np.load(f) for f in dust_files])
dust = np.load(dust_files[3])
print(dust.shape)
fig, ax = plt.subplots()
#ax.imshow(np.flipud(dust), vmin=0, vmax=2, cmap='cividis')
ax.imshow(dust, vmin=0, vmax=1, cmap='viridis')
ax.set_yticks([220, 225, 230])
plt.savefig('/home/kyle/Downloads/testd.png', dpi=200)

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

'''# Plot parameters
fig, ax = plt.subplots(1, 5, figsize=(10, 2))
dustvmax = 2
icevmax = 0.6
errorvmax = 0.1
latmin = -45
latmax = 45
lonmin = 180#+45
lonmax = 270#+45
dustcmap = 'cividis'
icecmap = 'viridis'
errorcmap = 'magma'

lt = ['11:25', '13:01', '13:27', '14:42', '15:31']

#######################
### Add in the IUVS data in QL form
#######################
for c, orbit in enumerate([3469, 3486, 3475, 3464, 3453]):
    print(orbit)
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
    swath_num = swath_number(fov)


    def make_swath_grid(field_of_view: np.ndarray, swath_num: int,
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
        slit_angles = np.linspace(angular_slit_width * swath_num,
                                  angular_slit_width * (swath_num + 1),
                                  num=n_positions+1)
        mean_angle_difference = np.mean(np.diff(field_of_view))
        field_of_view = np.linspace(field_of_view[0] - mean_angle_difference / 2,
                                    field_of_view[-1] + mean_angle_difference / 2,
                                    num=n_integrations + 1)
        return np.meshgrid(slit_angles, field_of_view)

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

    for swath in np.unique(swath_num):
        x, y = latlon_meshgrid(lat[swath==swath_num], lon[swath==swath_num], alt[swath==swath_num])
        ciax = ax[c].pcolormesh(x, y, dust[swath==swath_num], vmin=0, vmax=dustvmax, cmap='cividis')

    ax[c].set_xlim(lonmin, lonmax)
    ax[c].set_ylim(latmin, latmax)
    ax[c].set_facecolor('gray')
    if c != 0:
        ax[c].set_yticks([])
        ax[c].set_xticks([])
    ax[c].set_title(lt[c])

    if c == 4:
        divider = make_axes_locatable(ax[c])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(ciax, cax=cax, orientation='vertical')


plt.savefig('/home/kyle/Downloads/diurnaldust.png', dpi=200)'''
