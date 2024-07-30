import numpy as np
import matplotlib.pyplot as plt

c0g = np.loadtxt('catalogs/wcs_check/band0_x_hscg_catalog.csv', delimiter=',', skiprows=1)
c0r = np.loadtxt('catalogs/wcs_check/band0_x_hscr_catalog.csv', delimiter=',', skiprows=1)
c1g = np.loadtxt('catalogs/wcs_check/band1_x_hscg_catalog.csv',delimiter=',', skiprows=1)
c1r = np.loadtxt('catalogs/wcs_check/band1_x_hscr_catalog.csv', delimiter=',',skiprows=1)
c2g = np.loadtxt('catalogs/wcs_check/band2_x_hscg_catalog.csv', delimiter=',',skiprows=1)
c2r = np.loadtxt('catalogs/wcs_check/band2_x_hscr_catalog.csv', delimiter=',',skiprows=1)

arr = [c0g, c0r, c1g, c1r, c2g, c2r]

cols = []

max_dist = 0.0141
for obj in c2g:
    val = ( (obj[0]-obj[1])**2 + (obj[2]-obj[3])**2 )**.5
    cols.append(val/max_dist)

plt.scatter(c2g[:,0], c2g[:,2], c=cols, cmap='viridis', s=10)
plt.xlabel('BIT alpha')
plt.ylabel('BIT delta')
plt.colorbar(label='Rel. distance between BIT and HSC object')
plt.title('Distances between catalogued objects')
plt.savefig('catalogs/wcs_check/2xg_map.png')
plt.show()
