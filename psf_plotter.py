# produces the f-value calibration for interpreting HSC flux values (see report)

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS

# median of pixel values on far disk around point - estimate of background
def get_background(ra, dec):
    x, y = wcs.wcs_world2pix(ra,dec, 1) #appears that this .fits file uses the 1-based origin
    rad = 24
    pix_vals = []
    #sweeps pixel vals on boundary (in theory can be more accurate if divided smaller)
    div = 32
    for i in range(div):
        angle = 2*np.pi*i/div
        val = data[int(y + rad*np.sin(angle)), int(x + rad*np.cos(angle))] #data[y,x]
        pix_vals.append(val)
    pix_vals = np.array(pix_vals)
    return np.median(pix_vals)

# sum of background-subtracted pixel values in 12-pixel radius circle around point
def integrate_area(ra,dec):
    rad = 12
    bck = get_background(ra,dec) # assumes same bck for all pixels in object (SE chooses this too for meshes))
    x,y = wcs.wcs_world2pix(ra,dec,1) # center of object pixel
    
    #checks what pixels lie in circle, gets values, and subtracts background
    sum_vals = 0
    for i in range(rad):
        for j in range(rad):
            if (i**2 + j**2)<rad:
                #circle symmetry
                sum_vals = sum_vals + data[int(y+j), int(x+i)] + data[int(y-j), int(x+i)] + data[int(y+j), int(x-i)] +data[int(y-j), int(x-i)] - 4*bck
    return sum_vals

# sum of PSF's for all pixels in 12 pix radius around object center
def get_f(ra, dec):
    x,y = wcs.wcs_world2pix(ra,dec,1)
    d_0 = data[int(y), int(x)] - get_background(ra, dec)  # pixel value at obj barycenter, minus background
    int_ar = integrate_area(ra,dec) # pixel vals integrated over area, minus background
    f = int_ar/d_0
    return f

'''
# takes array containing [ra,dec] as entries to return avg f
def avg_f(arr):
    f_vals = []
    for star in arr:
        f_vals.append(get_f(star[0],star[1]))
    f_vals = np.array(f_vals)
    return np.mean(f_vals)
'''

#HSC Section

hsc_cat = np.loadtxt('cat_files/hscg.cat', skiprows = 1)
hsc_pix_to_arcsec = 3600 * 4.6666666666640*(10**-5) 

#ra and dec of stars (point sources), cutting to try to only get stars
hsc_star_cal = []
for obj in hsc_cat:
    in_box = obj[9]<197.9150 and obj[9]>197.8255 and obj[10]<-1.210 and obj[10]>-1.475
    # in common region and with star semi-major axis
    if in_box and hsc_pix_to_arcsec*obj[11]>.22 and hsc_pix_to_arcsec*obj[11]<.24: # angular resolution of HSC telescope, stars believed to lie here
        hsc_star_cal.append([obj[9], obj[10]])

#initializing astropy methods
with fits.open('fits_files/cutout-HSC-G-9360-pdr3_wide-231122-180146.fits') as hdul:
    hdul.info()
    header = hdul[1].header
    data = hdul[1].data
wcs = WCS(header)

f_vals = []
pix_vals = []
for star in hsc_star_cal:
    x,y = wcs.wcs_world2pix(star[0], star[1] ,1)
    cut = .1 # minimum brightness
    if data[int(y),int(x)] > cut:
        pix_vals.append(data[int(y), int(x)])
        f_vals.append(get_f(star[0],star[1]))
f_vals = np.array(f_vals)

print('HSC data:')
print('f value statistics of ' + str(f_vals.size) + ' stars considered (brightness > ' + str(cut) + '):')
print('Mean: ' + str(np.mean(f_vals)))
print('Median: ' + str(np.median(f_vals)))
print('Standard deviation: ' + str(np.std(f_vals)))
plt.hist(f_vals, bins = 40, range = (0,100))
plt.xlabel('f value of star')
plt.ylabel('Frequency')
plt.title('f values of stars (cut on star brightness)')
#plt.savefig('calib/hscg_cut_stars_f_hist.png')
plt.clf()

plt.scatter(pix_vals, f_vals)
plt.xlabel('Pixel value at star barycenter')
plt.ylabel('f value of star')
plt.title('Star brightness vs. star f value (cut on star brightness)')
#plt.savefig('calib/hscg_cut_stars_bright_v_f.png')
plt.clf()

#SuperBIT Section - wasn't officially used, with improvement could reveal info about SuperBIT PSF, yet have constructed curve
'''
bit_band1_cat = np.loadtxt('cat_files/bit_band1.cat', skiprows = 1)
bit_pix_to_arcsec = 0.140843

#ra and dec of stars (point sources)
bit_star_cal = []
for obj in hsc_cat:
    in_box = obj[9]<197.9150 and obj[9]>197.8255 and obj[10]<-1.210 and obj[10]>-1.475
    # in common region and with star semi-major axis
    if in_box and bit_pix_to_arcsec*obj[11]>.29 and bit_pix_to_arcsec*obj[11]<.311:
        bit_star_cal.append([obj[9], obj[10]])

with fits.open('fits_files/a1689_band1_wcs.fits') as hdul:
    hdul.info()
    header = hdul[0].header
    data = hdul[0].data

wcs = WCS(header)

f_vals = []
pix_vals = []
for star in bit_star_cal:
    x,y = wcs.wcs_world2pix(star[0], star[1] ,1)
    low_cut = 0.00078
    high_cut = 0.001
    if data[int(y),int(x)] > low_cut and data[int(y), int(x)] < high_cut:
        f = get_f(star[0],star[1])
        if f > 0:
            pix_vals.append(data[int(y), int(x)])
            f_vals.append(f)

print('SuperBIT data')
f_vals = np.array(f_vals)
print('f value statistics of ' + str(f_vals.size) + ' stars considered (brightness > ' + str(cut) + '):')
print('Mean: ' + str(np.mean(f_vals)))
print('Median: ' + str(np.median(f_vals)))
print('Standard deviation: ' + str(np.std(f_vals)))
plt.hist(f_vals, bins = 40)
plt.xlabel('f value of star')
plt.ylabel('Frequency')
plt.title('f values of stars (cut on star brightness)')
#plt.savefig('calib/bit_band1_cut_stars_f_hist.png')
plt.show()


plt.scatter(pix_vals, f_vals)
plt.xlabel('Pixel value at star barycenter')
plt.ylabel('f value of star')
plt.title('Star brightness vs. star f value (cut on star brightness)')
#plt.savefig('calib/bit_band1_cut_stars_bright_v_f.png')
plt.show()
'''


