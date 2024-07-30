import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

#band0 = np.loadtxt("cat_files/bit_band0.cat", skiprows=1)
#band1 = np.loadtxt("cat_files/bit_band1.cat", skiprows=1)
#band2 = np.loadtxt("cat_files/bit_band2.cat", skiprows=1)

#ICRS

b0_tl = (198.0783279, -1.4250203)
b0_tr = (197.8714477, -1.1136779)
b0_bl = (197.8717591, -1.5615085)
b0_br = (197.6665344, -1.2505200)

b1_tl = (198.0248082, -1.4506310)
b1_tr = (197.8870619, -1.1383359)
b1_bl = (197.8606218, -1.5477910)
b1_br = (197.6864116, -1.2361303)

b2_tl = (197.9793269, -1.4902198)
b2_tr = (197.9326814, -1.2055929)
b2_bl = (197.8353275, -1.5108104)
b2_br = (197.7410555, -1.2006205)

hsc_tl = (197.993518, -1.1827902)
hsc_tr = (197.7676478, -1.1839793)
hsc_bl = (197.999518, -1.4784152)
hsc_br = (197.7694611, -1.4795915)

com_tl = (197.8255, -1.210)
com_tr = (197.9150, -1.210)
com_br = (197.9150, -1.475)
com_bl = (197.8255, -1.475)

b0_gon = Polygon([b0_tl, b0_tr, b0_br, b0_bl], closed=True, edgecolor='y',facecolor='none', label='bit band 0')
b1_gon = Polygon([b1_tl, b1_tr, b1_br, b1_bl], closed=True, edgecolor='g',facecolor='none', label = 'bit band 1')
b2_gon = Polygon([b2_tl, b2_tr, b2_br, b2_bl], closed=True,edgecolor='b', facecolor='none', label = 'bit band 2')
hsc_gon = Polygon([hsc_tl, hsc_tr, hsc_br, hsc_bl], closed=True, edgecolor='r',facecolor='none', label= 'hsc bands')
com_gon = Polygon([com_tl, com_tr, com_br, com_bl], closed=True, edgecolor='k', facecolor='0.8', label = 'common field')

fig,ax = plt.subplots()
ax.add_patch(b0_gon)
ax.add_patch(b1_gon)
ax.add_patch(b2_gon)
ax.add_patch(hsc_gon)
ax.add_patch(com_gon)
ax.legend()

ax.set_xlim(197.6, 198.2)
ax.set_ylim(-1.7,-1)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('alpha (ICRS)')
plt.ylabel('delta (ICRS)')
plt.title('Observation Fields of Images')
plt.grid(True)

plt.savefig('common_box.png')



