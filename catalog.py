# Produces .csv files of objects commonly identified in given SuperBIT and HSC bands

import numpy as np
import os

bit_data=np.loadtxt("a1689/cat_files/bit_band2.cat", skiprows=1)
hsc_data=np.loadtxt("a1689/cat_files/hscg.cat", skiprows=1)

matches = []

thresh = (2*10**-4) # was 5*10**-4
#thresh = 10**-2 (for WCS error check)

# set common window of consideration (197 is a1689, 150 is cosmosk)
alpha_max = 197.9150 #150.300
alpha_min = 197.8255 #150.155
delta_max = -1.210 #2.345
delta_min = -1.475 #2.100

bit = []
hsc = []

# filter to only include common window
for obj in bit_data:
    if obj[11] < alpha_max and obj[11] > alpha_min and obj[12] < delta_max and obj[12] > delta_min:
        bit.append([obj[0], obj[1], obj[2], obj[3], obj[4], obj[5], obj[6], obj[7], obj[8], obj[9], obj[10], obj[11], obj[12], obj[13], obj[14]])
for obj in hsc_data:
    if obj[11] < alpha_max and obj[11] > alpha_min and obj[12] < delta_max and obj[12] > delta_min:
        hsc.append([obj[0], obj[1], obj[2], obj[3], obj[4], obj[5], obj[6], obj[7], obj[8], obj[9], obj[10], obj[11], obj[12], obj[13], obj[14]])

bit = np.array(bit)
hsc = np.array(hsc)

# check if objects close enough in distance
for bit_obj in bit:
    for hsc_obj in hsc:
        #sufficiently close to match
        match_alpha_sky = np.abs(bit_obj[11]-hsc_obj[11])<thresh
        match_delta_sky = np.abs(bit_obj[12]-hsc_obj[12])<thresh
        if match_alpha_sky and match_delta_sky:
            
            arr = [bit_obj[1], hsc_obj[1], bit_obj[2], hsc_obj[2], bit_obj[3], hsc_obj[3], bit_obj[4], hsc_obj[4], bit_obj[5], hsc_obj[5], bit_obj[6], hsc_obj[6], bit_obj[7], hsc_obj[7], bit_obj[8], hsc_obj[8], bit_obj[9], hsc_obj[9], bit_obj[10], hsc_obj[10], bit_obj[11], hsc_obj[11], bit_obj[12], hsc_obj[12], bit_obj[13], hsc_obj[13], bit_obj[14], hsc_obj[14]]
            #arr = [bit_obj[9], hsc_obj[9], bit_obj[10], hsc_obj[10], bit_obj[11], hsc_obj[11], bit_obj[12], hsc_obj[12]]
            matches.append(arr)

matches = np.array(matches)

column_labels = ['bit_flux_iso', 'hsc_flux_iso', 'bit_iso_fluxerr', 'hsc_iso_fluxerr', 'bit_flux_kron', 'hsc_flux_kron', 'bit_kron_fluxerr', 'hsc_kron_fluxerr','bit_background', 'hsc_background', 'bit_thresh', 'hsc_thresh','bit_maxflux', 'hsc_maxflux', 'bit_isoarea', 'hsc_isoarea','bit_xworld', 'hsc_xworld', 'bit_yworld', 'hsc_yworld', 'bit_alpha', 'hsc_alpha', 'bit_delta', 'hsc_delta', 'bit_a', 'hsc_a', 'bit_b', 'hsc_b']
#column_labels = ['bit_alpha', 'hsc_alpha', 'bit_delta', 'hsc_delta', 'bit_a', 'hsc_a', 'bit_b', 'hsc_b']

np.savetxt('a1689/catalogs/band2_x_hscg_catalog.csv', matches, delimiter=',', header=','.join(column_labels), comments='')

os.system('echo -e "\a"') # can take a while, alerts you

