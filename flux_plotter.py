# Flux Plotter - the central script for the data analysis
# Generates and imposes cuts on data, makes plots, and runs MCMC analysis

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from scipy.optimize import minimize
import emcee
from scipy import stats
import corner

#arcsec/pix conversion - make sure to change this throughout the code
bit_a1689_scale = .140843
bit_cosmos_scale = .140732


# initializing values and methods
data = np.loadtxt('a1689/catalogs/band1_x_hscg_catalog.csv', delimiter = ',', skiprows=1) # change the path for different datasets
psf = np.loadtxt("a1689/flux_plots/psf_cut.txt") # this pulls from the PSF-like form's file
flux_med = np.median(data[:,12])
flux_mean = np.mean(data[:,12])


# curve fitting to generate the PSF-generated curve to eliminate increasing brightness effects (see report)
def psf_form(x, a,b):
    return a*np.power(x,b)

def psf_func(x, a, b, c):
    return a*psf_form(b*x+c, popt[0], popt[1])

def model(x,a,b,c,d,e):
    return a*(b**(c*(x+d)))+ e

# collecting objects that are likely to be influenced by this phenomenon
pairs = [] # uses peak flux
for obj in data:
    if (obj[12]>flux_med and bit_a1689_scale*obj[24]<.4):
        pairs.append([bit_a1689_scale*obj[24],obj[12]])
pairs = np.array(pairs)

# different model fits to try to approximate this curve on this dataset
popt,pcov = curve_fit(psf_form, psf[:,0],psf[:,1], maxfev=100000) # runs from the imported data
pop, pcov = curve_fit(psf_func, pairs[:,0], pairs[:,1], p0 = [1, 1, 1], bounds = ([0,.0001, .001], [100, 100, 100]), maxfev=100000) # distorts the imported curve to fit data
mod, mvar = curve_fit(model, pairs[:,0], pairs[:,1], p0 = [1, 5, 1, 1,0], bounds = ([0,.0001, .001, -100,-10], [100, 100, 100, 100,100]), maxfev=1000000) # doesn't use imported data, simply tries to fit


# plots of data with cuts drawn
plt.scatter(bit_cosmos_scale*data[:,24], data[:,12], color = 'blue', label = 'Non-curve-fitting data')
plt.scatter(pairs[:,0], pairs[:,1], color = 'black', label = 'Curve-fitting data')
plt.scatter(bit_cosmos_scale*data[:,24], model(bit_cosmos_scale*data[:,24], mod[0], mod[1], mod[2], mod[3], mod[4]), color = 'orange', label =  'Exp fit cut')
plt.scatter(bit_cosmos_scale*data[:,24], psf_func(bit_cosmos_scale*data[:,24], pop[0], pop[1], pop[2]), color = 'red', label = "Jones fit cut")
plt.axhline(y=flux_med, color = 'green', linestyle = '--', label = 'Median: ' + str(flux_med))
plt.axhline(y=flux_mean, color = 'purple', linestyle = '--', label = 'Mean: ' + str(flux_mean))
plt.xlabel('Angular Size (arcseconds)')
plt.ylabel('Peak Flux')
plt.ylim(0,10)
plt.title('SuperBIT: Size vs. Peak Flux with cuts')
plt.legend()
#plt.savefig('cosmosk/flux_plots/kron/cuts/cuts_band1_hscg.png')
plt.show()
plt.clf()


# subjects data to cuts
bit_flux= []
hsc_flux = []
coords = []
ang = []
bit_back = []
errs = []
manual_min_bit = 0.005 # 0.005 for trying to eliminate very dim objects
manual_max_bit = .07 # .07 for trying to get same regime for cosmos and a1689, otherwise can be higher
manual_max_hsc = 15000 # unnecessary, but helpful to have parameter in case

# data must pass these cuts and be below the fitted curve from above
for obj in data:
    if obj[4] > manual_min_bit and obj[4] < manual_max_bit and obj[5]< manual_max_hsc and obj[12] < model(bit_a1689_scale*obj[24], mod[0], mod[1], mod[2], mod[3],mod[4]) and obj[12]<flux_mean and obj[4]/obj[8]>10: #if no saturation and falls under constructed psf curve (and obj[8] < flux_mean)
        bit_flux.append(obj[4])
        hsc_flux.append(obj[5])
        coords.append([obj[20],obj[22]])
        ang.append(bit_a1689_scale*obj[24])
        errs.append([obj[6],obj[7]])
        bit_back.append(obj[8])


bit_flux = np.array(bit_flux)
hsc_flux = np.array(hsc_flux)
coords = np.array(coords)
ang = np.array(ang)
bit_back = np.array(bit_back)
errs = np.array(errs)



#outlier exclusion - binary vals for every data point, then removes 0's
binary = np.ones(coords[:,0].size)
outlier_list = np.loadtxt('a1689/flux_plots/outlier_files/c1g_outliers.csv', delimiter=',') # created from inspecting with DS9

for i in range(coords[:,0].size):
    alpha = coords[i,0]
    delta = coords[i,1]
    for j in range(outlier_list[:,0].size):
        if alpha == outlier_list[j,0] and delta == outlier_list[j,1]:
            binary[i]=0

b_=[]
h_=[]
a_=[]
bb_=[]
er_=[]
co_=[]

# slicing
for i in range(binary.size):
    if binary[i] == 1:
        b_.append(bit_flux[i])
        h_.append(hsc_flux[i])
        a_.append(ang[i])
        co_.append(coords[i])
        bb_.append(bit_back[i])
        er_.append(errs[i])

bit_flux = np.array(b_)
hsc_flux = np.array(h_)
ang = np.array(a_)
bit_back = np.array(bb_)
errs = np.array(er_)
coords = np.array(co_)

ratios = (hsc_flux/bit_flux)

# makes region with all data that will be considered for analysis
with open('a1689/region_overlays/MCMC_data/2r_MCMC.reg', 'w') as f:
    f.write('icrs\n')
    for i in range(bit_flux.size):
        f.write('point(' + str(coords[i,0]) +',' +str(coords[i,1]) + ')\n')


# S/N plots
plt.hist(bit_flux/bit_back, density = True, bins = 20)
plt.xlabel('S/N (Kron Flux above Background/Background)')
plt.ylabel('Frequency')
#plt.xlim(0,10000)
plt.title('SuperBIT Signal to Noise on Selection')
plt.figtext(1.0,0.2, pd.DataFrame(bit_flux/bit_back).describe().to_string())
#plt.savefig('cosmosk/flux_plots/kron/sig_and_ang/sig_noise_band2_hscr.png', bbox_inches='tight')
plt.show()
plt.clf()

# ang plots
plt.hist(ang, density = True, bins = 20)
plt.xlabel('Angular Size - BIT Semi-Major Axis (arcseconds)')
plt.ylabel('Frequency')
plt.title('Angular Size of BIT Objects')
plt.figtext(1.0,0.2, pd.DataFrame(ang).describe().to_string())
#plt.savefig('cosmosk/flux_plots/kron/sig_and_ang/ang_band2_hscr.png', bbox_inches = 'tight')
plt.show()
plt.clf()

# plot of processed data
plt.scatter(ang, bit_flux)
plt.xlabel('Angular Size (arcseconds)')
plt.ylabel('Kron Flux')
plt.title('SuperBIT Kron Flux v. Angular Size (cut)')
#plt.savefig('cosmosk/flux_plots/kron/flux_v_size/flux_v_size_band2_hscr.png')
plt.show()
plt.clf()

'''
# AB mag conversion for HSC, denominator is MAGZERO which may change, 27.516 is mean f-value (see report)
#for i in range(hsc_flux.size):
#    hsc_flux[i] = -2.5*np.log10(27.516953*hsc_flux[i]/63095734448.01944) #HSC mag conversion

'''

# percent data lying in interval around median
med = np.median(ratios)
margin = 1950
count = 0
for entry in ratios:
    if entry < med + margin and entry > med - margin:
        count = count + 1
print("Approximately " + str(count/ratios.size) + '% of the data lies within +- ' + str(margin) ' of the median (' + str(med) + ')')

# region generator for ratio percentile outliers
low = np.percentile(ratios, 10)
high = np.percentile(ratios, 90)
with open('region_overlays/ratio_outliers/band0_x_hscg_outliers.reg', 'w') as f:
    f.write('icrs\n')
    for obj in data:
        if obj[0] < manual_cut and obj[8] < flux_mean and obj[8] < model(bit_cosmos_scale*obj[20], mod[0], mod[1], mod[2], mod[3],mod[4]): #if no saturation and falls under constructed psf curve
            if obj[1]/obj[0] > high or obj[1]/obj[0]<low:
                f.write('point(' + str(obj[16]) +',' +str(obj[18]) + ')\n')

# ratios histogram
plt.hist(ratios, bins = 15, density = True)
#plt.xscale('log')
plt.title('kron_flux_hscr/kron_flux_bitband2')
plt.xlabel('Kron Flux Ratio (non-magnitude)')
plt.ylabel('Frequency')
plt.figtext(1.0,0.2, pd.DataFrame(ratios).describe().to_string())
#plt.savefig('cosmosk/flux_plots/kron/ratios/ratio_band2_hscr.png', bbox_inches='tight')
plt.show()
plt.clf()


#Markov Chain Monte Carlo Analysis

# log-likelihood for 2-d uncertainty orthogonal regression model
def log_likelihood(params):
    m,b = params
    theta = np.arctan(m)
    total = 0
    for i in range(bit_flux.size):
        x = bit_flux[i]
        y = hsc_flux[i]
        x_var = errs[i,0]**2
        y_var = errs[i,1]**2
        total = total - ((-x*np.sin(theta)+(y-b) *np.cos(theta))**2/(x_var*(np.sin(theta)**2)+y_var*(np.cos(theta)**2)))
    return total/2

# priors (make sure to change!)
def log_prior(params):
    m,b = params
    if m > 6000 and m < 10000 and b > -100 and b < 100:
        return 0.0
    else:
        return -np.inf

def log_probability(params):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + log_likelihood(params)


# initialising parameters and starting values
ndim = 2
nwalkers = 30
nsteps = 2000
initial_state = 10*np.random.randn(nwalkers,ndim)

# initializing the guesses
m_guess = 8000
for i in range(nwalkers):
    initial_state[i,0] = initial_state[i,0] + m_guess

# running MCMC
sampler = emcee.EnsembleSampler(nwalkers,ndim,log_probability, a = .1)
sampler.run_mcmc(initial_state,nsteps,progress=True)
samples = sampler.get_chain()

# time series plots
fig, axes = plt.subplots(2,figsize=(10,7),sharex=True)
labels = ['m','b']
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:,:,i], 'k', alpha=.3)
    ax.set_xlim(0,len(samples))
    ax.set_ylabel(labels[i])
axes[-1].set_xlabel('Step Number')
#plt.savefig('cosmosk/flux_plots/kron/MCMC/2r_param_steps.png')
plt.show()

# cleaning data, turning to flat data 
burn_in = 200
thinning = 30
flat_samples = sampler.get_chain(discard = burn_in, thin = thinning, flat = True)
np.savetxt('cosmosk/flux_plots/kron/MCMC/2rMCMC.csv', flat_samples, delimiter=',')

mean_m, mean_b = np.mean(flat_samples, axis=0) # slope and intercept means
std_m, std_b = np.std(flat_samples, axis=0) # raw slope and intercept variances

# triangle plot
fig = corner.corner(flat_samples, labels = labels, truths=[4233,0])
axes = np.array(fig.axes).reshape((ndim,ndim))
axes[0,1].text(.96, .96,
        'm: μ = ' + str(round(mean_m, 2)) + ', σ = ' + str(round(std_m, 2)) + '\nb: μ = ' + str(round(mean_b,2)) + ', σ = ' + str(round(std_b,2)),
        transform = axes[0,1].transAxes, horizontalalignment = 'right', verticalalignment = 'top', fontsize=12, bbox=(dict(facecolor='white', alpha=.8))
)
#plt.savefig('cosmosk/flux_plots/kron/MCMC/1g_triang.png')
plt.show()


# z-scores of points wrt line
theta = np.arctan(mean_m)
zs = []
for i in range(bit_flux.size):
    d = -np.sin(theta)*bit_flux[i] + np.cos(theta)*hsc_flux[i] - mean_b*np.cos(theta)
    cov = (errs[i,0]*np.sin(theta))**2 + (errs[i,1]*np.cos(theta))**2
    zs.append(d/np.sqrt(cov))
plt.hist(zs, density = True)
plt.xlabel("Point Z-Score from Best-Fit Line")
plt.ylabel("Frequency")
plt.title("Z-Scores of Data Points from Orthogonal Regression")
#plt.savefig('cosmosk/flux_plots/kron/MCMC/zs_2r.png')
plt.show()


# regression plot
plt.plot(bit_flux,bit_flux*mean_m + mean_b, color = 'blue', label = 'Best fit: hsc = ' + str(round(mean_m, 2)) + "*bit + " + str(round(mean_b,2)))
plt.scatter(bit_flux,hsc_flux)
plt.xlabel('SuperBIT Kron Flux')
plt.ylabel('HSC Kron Flux (non-magnitude)')
plt.title('SuperBIT and HSC Kron Fluxes, MCMC Fit')
plt.legend()
#plt.savefig('cosmosk/flux_plots/kron/compares/compare_band2_hscr.png')
plt.show()
plt.clf()

# residuals plot
plt.scatter(bit_flux, bit_flux*mean_m + mean_b - hsc_flux)
plt.axhline(y = 0, color = 'green', linestyle='--', label = 'Residual = 0')
plt.xlabel('SuperBIT Kron Flux')
plt.ylabel('Residual from BIT->HSC Regression')
plt.title('MCMC Residual Plot of hsc = ' + str(round(mean_m, 2)) + "*bit + " + str(round(mean_b,2)))
plt.legend()
#plt.savefig('cosmosk/flux_plots/kron/compares/residuals_band2_hscr.png')
plt.show()

# high residual region generator
with open('cosmosk/region_overlays/outliers/band1_x_hscg_outliers.reg', 'w') as f:
    f.write('icrs\n')
    for i in range(bit_flux.size):
        if np.abs(zs[i])>4:
            f.write('point(' + str(coords[i,0]) +',' +str(coords[i,1]) + ')\n')
