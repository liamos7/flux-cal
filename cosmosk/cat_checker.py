import numpy as np



alpha_max = 150.300
alpha_min = 150.155
delta_max = 2.345
delta_min = 2.100


cat0 = np.loadtxt('cat_files/bit_band0.cat', skiprows=1)
cat1 = np.loadtxt('cat_files/bit_band1.cat', skiprows=1)
cat2 = np.loadtxt('cat_files/bit_band2.cat', skiprows=1)
catg = np.loadtxt('cat_files/hscg.cat', skiprows=1)
catr = np.loadtxt('cat_files/hscr.cat', skiprows=1)


match0g = np.loadtxt('catalogs/band0_x_hscg_catalog.csv', delimiter=',', skiprows=1)
match0r = np.loadtxt('catalogs/band0_x_hscr_catalog.csv', delimiter=',', skiprows=1)
match1g = np.loadtxt('catalogs/band1_x_hscg_catalog.csv',delimiter=',', skiprows=1)
match1r = np.loadtxt('catalogs/band1_x_hscr_catalog.csv', delimiter=',',skiprows=1)
match2g = np.loadtxt('catalogs/band2_x_hscg_catalog.csv', delimiter=',',skiprows=1)
match2r = np.loadtxt('catalogs/band2_x_hscr_catalog.csv', delimiter=',',skiprows=1)


cats = [cat0, cat1, cat2, catg, catr]

print(cat0)
matches = [match0g, match0r, match1g, match1r, match2g, match2r]
print(match0g)

c = [0,0,0,0,0]
m = [0,0,0,0,0,0]

for i in range(len(cats)):
    for obj in cats[i]:
        if obj[9] < alpha_max and obj[9] > alpha_min and obj[10] < delta_max and obj[10] > delta_min:
            c[i] = c[i]+1


for i in range(len(matches)):
    for obj in matches[i]:
        if obj[16] < alpha_max and obj[16] > alpha_min and obj[18] < delta_max and obj[18] > delta_min:
            m[i] = m[i]+1


print(m)
print(c)


