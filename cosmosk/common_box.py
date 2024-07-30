import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


b0_tl = (150.3351495, 2.0934521)
b0_tr = (150.2980656, 2.4127832)
b0_bl = (150.1533485, 2.0743552)
b0_br = (150.1006952, 2.3897640)

b1_tl = (150.3677376, 2.0914540)
b1_tr = (150.2877092, 2.4121069)
b1_bl = (150.1445241, 2.0450391)
b1_br = (150.0869719, 2.3856112)

b2_tl = (150.3607400, 2.0979915)
b2_tr = (150.2874751, 2.4141759)
b2_bl = (150.1532408, 2.0470617)
b2_br = (150.0925110, 2.3830881)

hsc_tl = (150.3377872, 2.3485827)
hsc_tr = (149.9312159, 2.3485497)
hsc_bl = (150.3377697, 2.0616111)
hsc_br = (149.9325211, 2.0603402)

com_tl = (150.155, 2.345)
com_tr = (150.300, 2.345)
com_br = (150.300, 2.100)
com_bl = (150.155, 2.100)

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

ax.set_xlim(149.75, 150.4)
ax.set_ylim(2,2.5)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('alpha (ICRS)')
plt.ylabel('delta (ICRS)')
plt.title('Observation Fields of Images')
plt.grid(True)
plt.savefig('common_box.png')



