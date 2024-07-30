import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt("cosmosk/cat_files/hscg.cat", skiprows=1)

bit_a1689_scale = .140843
bit_cosmos_scale = .140732

alpha_max = 150.300 #197.9150 
alpha_min = 150.155 #197.8255
delta_max = 2.345 #-1.210
delta_min = 2.100 #-1.475

objs = []

print('Starting window cut')
for obj in data:
    if obj[11] < alpha_max and obj[11] > alpha_min and obj[12] < delta_max and obj[12] > delta_min:
        objs.append([obj[0], obj[1], obj[2], obj[3], obj[4], obj[5], obj[6], obj[7], obj[8], obj[9], obj[10], obj[11], obj[12], obj[13], obj[14]])
print('Window cut completed')
flux = []
coords = []
ang = []
bit_back = []
err = []

print('Starting to form parameter arrays')
for obj in objs:
    flux.append(obj[3])
    coords.append([obj[11],obj[12]])
    #ang.append(bit_cosmos_scale*obj[13])
    #err.append(obj[4])
    #bit_back.append(obj[5])

print('Parameter arrays completed')
flux = np.array(flux)
coords = np.array(coords)
ang = np.array(ang)
bit_back = np.array(bit_back)
err = np.array(err)


cut = np.mean(flux)
radius = 0.005
densities = []

print('Starting density checking')
for i in range(flux.size):
    print(str(round(i/flux.size,3)) + '%')
    count = 0
    ra = coords[i,0]
    dec = coords[i,1]
    for j in range(flux.size):
        if (flux[j] > cut):
            if np.sqrt((ra-coords[j,0])**2 + (dec-coords[j,1])**2) < radius:
                count = count + 1
    densities.append(count)

plt.hist(densities, density=False)
plt.xlabel('# of bright objects in radius ' + str(radius) + ' degrees about obj')
plt.ylabel('Frequency')
#plt.title('Neighbor Density (total)')
plt.title('Neighbor Density (Objects brighter than mean flux)')
plt.savefig('cosmosk/density_plots/neighbor/hscg_density_cut.png')
plt.show()



