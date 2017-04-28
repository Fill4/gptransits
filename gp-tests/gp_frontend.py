from config_file import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import pyfits
import timeit
import gp_backend

# Start timer
print("\nStarting GP fitting procedure")
print("-------------------------------------------------------\n")
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
gp_backend.run_minimization(data, priors, plot=plot)

# Run MCMC
gp_backend.run_mcmc(data, priors, plot=plot, nwalkers=16)

if plot:
	plt.show()

# Print execution time
fullTimeScript = timeit.default_timer() - startTimeScript
print("-------------------------------------------------------\n")
print("Complete execution time: {} usec".format(fullTimeScript))