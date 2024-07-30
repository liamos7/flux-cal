import numpy as np

cat_data = np.loadtxt('catalogs/band0_x_hscg_catalog.csv', delimiter=',', skiprows=1)
bit_data = np.loadtxt("cat_files/bit_band0.cat", skiprows=1)

matches = []
inside_region=0

thresh = 5*10**-5

for bit_obj in bit_data:
    match=False
    in_box = bit_obj[9]<197.9150 and bit_obj[9]>197.8255 and bit_obj[10]<-1.210 and bit_obj[10]>-1.475
    if in_box == True:
        inside_region = inside_region + 1
    for cat_obj in cat_data:
        match_x_world = np.abs(bit_obj[7]-cat_obj[12])<thresh
        match_y_world = np.abs(bit_obj[8]-cat_obj[14])<thresh
        match_alpha_sky = np.abs(bit_obj[9]-cat_obj[16])<thresh
        match_delta_sky = np.abs(bit_obj[10]-cat_obj[18])<thresh
        if match_x_world and match_y_world and match_alpha_sky and match_delta_sky:
            match = True

    #append if no matches in catalog
    if match == False and in_box == True:
        arr = [bit_obj[1], bit_obj[2], bit_obj[3], bit_obj[4], bit_obj[5], bit_obj[6], bit_obj[7], bit_obj[8], bit_obj[9], bit_obj[10], bit_obj[11], bit_obj[12]]
        matches.append(arr)

matches = np.array(matches)

column_labels = ['bit_flux_iso', 'bit_iso_fluxerr', 'bit_background','bit_thresh', 'bit_maxflux', 'bit_isoarea','bit_xworld', 'bit_yworld', 'bit_alpha', 'bit_delta', 'bit_a', 'bit_b']

np.savetxt('no_matches/band0_x_hscg_no_match.csv', matches, delimiter=',', header=','.join(column_labels), comments='')

print("# of objects in common area for SuperBIT Band: " + str(inside_region))
print("# of hits with HSC: " + str(cat_data.shape[0]))
print("# of no_match objects: " + str(matches.shape[0]))

