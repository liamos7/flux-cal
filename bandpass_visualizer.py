# Visualizes the bandpass distributions

import numpy as np
import matplotlib.pyplot as plt
import csv

# converts from .txt (how they appear online) to .csv (what we work with)
def to_csv(txt_name,csv_name):
    with open(txt_name, 'r') as txt_file, open(csv_name,'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for line in txt_file:
            columns = line.strip().split()
            if len(columns) == 2:
                csv_writer.writerow(columns)

to_csv('band_data/hscg_band.txt', 'band_data/hscg_band.csv')
to_csv('band_data/hscr_band.txt', 'band_data/hscr_band.csv')



#convert csv to array (SuperBIT has extra column so elim, convert ang to nano)
hsc_g =  np.loadtxt('band_data/hscg_band.csv', delimiter=',', skiprows=1)
hsc_r =  np.loadtxt('band_data/hscr_band.csv', delimiter=',', skiprows=1)
bit_b = np.loadtxt('band_data/b.csv', delimiter=',', skiprows=1)
bit_b = bit_b[:,1:]
bit_g = np.loadtxt('band_data/g.csv', delimiter=',', skiprows=1)
bit_g = bit_g[:,1:]
bit_lum = np.loadtxt('band_data/lum.csv', delimiter=',', skiprows=1)
bit_lum = bit_lum[:,1:]
bit_nir = np.loadtxt('band_data/nir.csv', delimiter=',', skiprows=1)
bit_nir = bit_nir[:,1:]
bit_r = np.loadtxt('band_data/r.csv', delimiter=',', skiprows=1)
bit_r = bit_r[:,1:]
bit_u = np.loadtxt('band_data/u.csv', delimiter=',', skiprows=1)
bit_u = bit_u[:,1:]

plt.plot(.1*hsc_g[:,0], hsc_g[:,1],'-.', color = 'black', label = "HSC g")
plt.plot(.1*hsc_r[:,0], hsc_r[:,1],'-.', color = 'red', label = "HSC r")
plt.plot(bit_b[:,0], bit_b[:,1],'-.', color = 'grey', label = "BIT b")
plt.plot(bit_g[:,0], bit_g[:,1], '-.', color = 'green', label = "BIT g")
plt.plot(bit_lum[:,0], bit_lum[:,1], '-.', color = 'blue', label = "BIT lum")
plt.plot(bit_nir[:,0], bit_nir[:,1], '-.',color = 'darkorchid', label = "BIT nir")
plt.plot(bit_r[:,0], bit_r[:,1], '-.', color = 'violet', label = "BIT r")
plt.plot(bit_u[:,0], bit_u[:,1],'-.', color = 'hotpink', label = "BIT u")
plt.legend()
plt.xlabel('Wavelength (nanometers)')
plt.ylabel('Transmission')
plt.title('SuperBIT and HSC Bandpass Distributions')
#plt.savefig('band_data/bandpass_plot.png')
plt.show()
