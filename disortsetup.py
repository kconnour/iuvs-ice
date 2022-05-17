from numpy import f2py
import os

path_to_disort = '/home/kyle/repos/paper3'
folder_name = 'disort4.0.99'
module_name = 'disort'

disort_source_dir = os.path.join(path_to_disort, folder_name)
mods = ['BDREF.f', 'DISOBRDF.f', 'ERRPACK.f', 'LAPACK.f',
        'LINPAK.f', 'RDI1MACH.f']
paths = [os.path.join(disort_source_dir, m) for m in mods]
with open(os.path.join(disort_source_dir, 'DISORT.f')) as mod:
    f2py.compile(mod.read(), modulename=module_name, extra_args=paths)