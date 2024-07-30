import numpy as np

# for any csv's that have RA and dec as columns
data = np.genfromtxt('cosmosk/catalogs/band0_x_hscr_catalog.csv', delimiter=',', skip_header=1)
alphas = data[:,16]
deltas = data[:,18]


'''
data = np.loadtxt('cosmosk/cat_files/hscr.cat',skiprows=1)
alphas = data[:,9]
deltas = data[:,10]
'''

# for name.reg in icrs
with open('cosmosk/region_overlays/cat_regions/band0_x_hscr_cat_regions.reg', 'w') as f:
    f.write('icrs\n')
    for i in range(alphas.size):
        f.write('point(' + str(alphas[i]) +',' +str(deltas[i]) + ')\n')

f.close()
