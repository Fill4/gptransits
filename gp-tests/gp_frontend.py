from config_file import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pyfits
import timeit
import sys
import gp_backend

# Start timer
gp_backend.verboseprint('\n{:_^60}'.format('Starting GP fitting procedure'))
startTimeScript = timeit.default_timer()

# Read data from fits and remove nans
hdulist = pyfits.open(filename)

ntime = getattr(hdulist[1].data, fits_options['time'])[:Nmax]
nflux = getattr(hdulist[1].data, fits_options['flux'])[:Nmax]
nerror = getattr(hdulist[1].data, fits_options['error'])[:Nmax]

ind = np.logical_and(~np.isnan(ntime), ~np.isnan(nflux))
time = ntime[ind]
flux = nflux[ind]
error = nerror[ind]
flux = flux - np.mean(flux)

# Bundle data in tuple for organisation
data = (time, flux, error)

# Initiate prior distributions according to options set by user
priors = gp_backend.setup_priors(prior_settings)

# Run minimization
gp_backend.run_minimization(data, priors, plot=plot, module=module)

# Run MCMC
gp_backend.run_mcmc(data, priors, plot=plot, nwalkers=nwalkers, burnin=burnin, iterations=iterations, module=module)

if plot:
	for i in plt.get_fignums():
		plt.figure(i)
		plt.savefig('Figures/figure%d.png' % i, dpi = 200)

# Print execution time
fullTimeScript = timeit.default_timer() - startTimeScript
gp_backend.verboseprint("\nComplete execution time: {:10.5} usec".format(fullTimeScript))
gp_backend.verboseprint('\n{:_^60}\n'.format('END'))