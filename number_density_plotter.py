# somehwat old and could use update (to make it work for cosmosk), produces a binned grid that shows the number of objects in each bin

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

data = np.loadtxt('cat_files/bit_band2.cat', skiprows=1)

counts = np.zeros((5,15))

#for every detected object
for obj in data:
	# in the common region
	in_box = obj[9] < 197.9150 and obj[9] > 197.8255 and obj[10] < -1.210 and obj[10] > -1.475
	if in_box:
		# getting which 1 arcmin^2 box it falls into (0, 1, 2, 3, 4) x (0, 1..., 14)
		alpha_arc = int(np.floor((obj[9]-197.8255)*60))
		delta_arc = int(np.floor((obj[10]+1.475)*60))
		# dim is 5.4x15, cutting off edge boxes since most are missing area
		if alpha_arc!=5 and delta_arc!=15:
			counts[alpha_arc, delta_arc] = counts[alpha_arc, delta_arc] + 1


cmap = plt.get_cmap('viridis')
norm=colors.Normalize(vmin=np.min(counts),vmax=np.max(counts))
fig,ax = plt.subplots()

for i in range(5): #alphas
	for j in range(15): #deltas
		square = plt.Rectangle((i, j), 1, 1, facecolor=cmap(norm(counts[i, j])), edgecolor='black')
		ax.add_patch(square)
ax.set_xlim(0,5)
ax.set_ylim(0,15)
ax.set_aspect('equal', adjustable='box')
plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, orientation='vertical', label='Number Density')
plt.xlabel('RA-11869.53 (arcminutes)')
plt.ylabel('Dec+88.5 (arcminutes)')
plt.title('Number Density of Stellar Objects for bit_band2')

plt.savefig("density_plots/density_bit_band2.png")
