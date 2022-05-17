from pathlib import Path
import numpy as np

p = sorted(Path('/home/kyle/iceradprop').glob('*'))


cext = np.zeros((13, 1568))
csca = np.zeros((13, 1568))
g = np.zeros((13, 1568))

for i in range(13):
    foo = np.genfromtxt(p[i], skip_header=2)
    w = foo[:, 0]
    cext[i, :] = foo[:, 1]
    csca[i, :] = foo[:, 2]
    g[i, :] = foo[:, 3]


np.save('/home/kyle/repos/paper3/radprop/mars_ice/particle_sizes.npy', np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 4, 5, 6, 8]))
np.save('/home/kyle/repos/paper3/radprop/mars_ice/wavelengths.npy', w)
np.save('/home/kyle/repos/paper3/radprop/mars_ice/extinction_cross_section.npy', cext)
np.save('/home/kyle/repos/paper3/radprop/mars_ice/scattering_cross_section.npy', csca)
np.save('/home/kyle/repos/paper3/radprop/mars_ice/asymmetry_parameter.npy', g)
