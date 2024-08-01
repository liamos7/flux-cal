# displayed overlayed corner plots for the MCMC data

import numpy as np
import corner
import matplotlib.pyplot as plt
import emcee


a1689_1g = np.loadtxt("a1689/flux_plots/kron/MCMC/1gMCMC.csv", delimiter=',')
a1689_2r = np.loadtxt("a1689/flux_plots/kron/MCMC/2rMCMC.csv", delimiter=',')
cosmosk_1g = np.loadtxt("cosmosk/flux_plots/kron/MCMC/1gMCMC.csv", delimiter=',')
cosmosk_2r = np.loadtxt("cosmosk/flux_plots/kron/MCMC/2rMCMC.csv", delimiter=',')

fig = corner.corner(a1689_1g, labels = ['m', 'b'], color='blue',alpha=.5, hist_kwargs={'label':'a1689'})
corner.corner(cosmosk_1g, fig=fig, color='red', alpha=.5, hist_kwargs={'label':'cosmosk'})
plt.plot([],[],color='blue')
plt.plot([],[],color='red')
plt.legend()
#plt.savefig('1g_triangles.png')
plt.show()

fig = corner.corner(a1689_2r, labels = ['m', 'b'], color='blue',alpha=.5, hist_kwargs={'label':'a1689'})
corner.corner(cosmosk_2r, fig=fig, color='red', alpha=.5, hist_kwargs={'label':'cosmosk'})
plt.plot([],[],color='blue')
plt.plot([],[],color='red')
plt.legend()
#plt.savefig('2r_triangles.png')
plt.show()
