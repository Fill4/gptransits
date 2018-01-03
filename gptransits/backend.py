''' Filipe Pereira - 2017
General procedure for fitting light curve data to a model and using
gaussian processes to fit the remaining stochastic noise
'''

import numpy as np
import matplotlib.pyplot as plt
import sys, os
import emcee
import corner
import timeit
import pyfits
import scipy.stats
import scipy.optimize as op
from config_file import verbose

# Gaussian processes libraries
import george
import celerite

# Define print function that only prints in verbose mode
if verbose:
	def vprint(*args):
		print(*args)
else:   
	vprint = lambda *a: None # do-nothing function

def read_data(filename, Nmax, offset, fits_options=None):
	_, ext = os.path.splitext(filename)
	
	if ext == '.fits':
		# Read data from fits and remove nans
		hdulist = pyfits.open(filename)

		ntime = getattr(hdulist[1].data, fits_options['time'])[offset:Nmax+offset]
		nflux = getattr(hdulist[1].data, fits_options['flux'])[offset:Nmax+offset]
		nerror = getattr(hdulist[1].data, fits_options['error'])[offset:Nmax+offset]

		ind = np.logical_and(~np.isnan(ntime), ~np.isnan(nflux), ~np.isnan(nerror))
		time = ntime[ind]
		time = time - time[0]
		flux = nflux[ind]/np.median(nflux[ind])
		error = nerror[ind]/np.median(nflux[ind])
		flux = (flux - 1)*1e6 #to ppm
		error = error*1e6 #to ppm

		return (time, flux, error)

	elif ext == '.dat':
		time, flux, error = np.loadtxt(filename, unpack=True)
		error = None
		return (time, flux, error)


#----------------------------------------------------------------------------------
# Returns a GP object from the celerite module with the provided parameters params
def setup_gp(params):
	scaled_params = scale_params(params)
	
	#S1, w1 = params
	#S1, w1, S2, w2 = params
	S1, w1, S_bump, Q_bump, w_bump = np.log(scaled_params)
	#S1, w1, S2, w2, S_bump, Q_bump, w_bump = np.log(scaled_params)

	Q = 1.0 / np.sqrt(2.0)
	kernel_1 = celerite.terms.SHOTerm(log_S0=S1, log_Q=np.log(Q), log_omega0=w1)
	kernel_1.freeze_parameter("log_Q")
	#kernel_2 = celerite.terms.SHOTerm(log_S0=S2, log_Q=np.log(Q), log_omega0=w2)
	#kernel_2.freeze_parameter("log_Q")
	kernel = kernel_1 #+ kernel_2 

	kernel += celerite.terms.SHOTerm(log_S0=S_bump, log_Q=Q_bump, log_omega0=w_bump)

	#kernel += celerite.terms.JitterTerm(log_sigma=jitter)

	gp = celerite.GP(kernel)
	return gp

def print_params(params, priors):
	for par in priors:
		vprint("{:16} {:1} {:1} {:10.4f}".format(priors[par][0], "-", "", params[par]))

# Scale parameters from Diamonds to celerite according to Foreman-Mackey 2017
def scale_params(params):
	qsi = np.sqrt(2/np.pi)
	xfreq = 2*np.pi

	#S1, w1 = params
	#S1, w1, S2, w2 = params
	S1, w1, S_bump, Q_bump, w_bump = params
	#S1, w1, S2, w2, S_bump, Q_bump, w_bump = params

	S1 = (S1**2 / w1) * (2/np.sqrt(np.pi))
	w1 = w1*xfreq
	#S2 = (S2**2 / w2)*np.sqrt(2)
	#w2 = w2*xfreq

	S_bump = S_bump / ((Q_bump**2) * qsi)
	w_bump = w_bump*xfreq

	#scaled_params = [S1, w1]
	#scaled_params = [S1, w1, S2, w2]
	scaled_params = [S1, w1, S_bump, Q_bump, w_bump]
	#scaled_params = [S1, w1, S2, w2, S_bump, Q_bump, w_bump]
	return scaled_params

# Creates the dictionary priors with the instantiated probability distributions for each parameter 
def setup_priors(prior_settings):
	priors = {}
	for par in prior_settings:
		priors[par] = [prior_settings[par][0], getattr(scipy.stats, 
			prior_settings[par][1])(loc=prior_settings[par][2], 
			scale=prior_settings[par][3]-prior_settings[par][2])]
	return priors

# Draws num number of samples from each of the parameters distribution
def sample_priors(priors, num = 1):
	sample = np.zeros([len(priors), num])
	for par in priors:
		sample[par] = priors[par][1].rvs(num)
	return sample


#-----------------------------------------------------------------------------------
# Defining functions to handle the Monte Carlo method calculations
# Calculates the value of the model for parameters params in the positions time
def model(params, time):
	return np.zeros(len(time))

# Defines the priors for all the parameters
def log_prior(params, priors):
	prior_sum = 0
	# Evaluate parameters according to priors. Return sum when working in log
	for par in priors:
		prior_sum += priors[par][1].logpdf(params[par])
	return prior_sum

# Calculates the lnlikelihood of the parameters params proposed. Can include model and
# prior calculation besides the gaussian process
def log_likelihood(params, time, flux, error, priors, gp):
	
	# Check priors and return if any parameter is outside prior
	lnprior = log_prior(params, priors)
	if not np.isfinite(lnprior):
		return -np.inf
	
	scaled_params = scale_params(params)
	gp.set_parameter_vector(np.log(scaled_params))
	lnlikelihood = gp.log_likelihood(flux - model(params, time))
	
	return lnprior + lnlikelihood

# Calculates the gradient of the lnlikelihood for the proposed parameters params
# This quantity is used in minimization methods such as scipy minimize
# Commented because it's unnecessary to the minimization right now
''' 
def lnLikeGrad(params, phase, flux, error, minimization=False):
	gp = setup_george(params)
	gp.compute(phase, error)
	return -gp.grad_lnlikelihood(flux, quiet=True)
'''


#--------------------------------------------------------------------------------
# MCMC and minimization functions
def run_mcmc(data, priors, plot_corner=True, plot_sample=False, init_params=None, nwalkers=20, iterations=2000):
	
	itime_mcmc = timeit.default_timer()

	time, flux, error = data
	# Draw samples from the prior distributions to have initial values for all the walkers
	init_params = sample_priors(priors, nwalkers)
	ndim = len(init_params)

	gp = setup_gp(init_params[:,0])
	#gp.compute(time/1e6, error)
	gp.compute(time/1e6)

	# Initiate the sampler
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=data + (priors, gp))

	# Burn-in
	vprint('Starting Burn-in ...')
	burnin_params, _, _ = sampler.run_mcmc(init_params.T, 500)
	sampler.reset()

	#Run the walkers
	vprint('Starting MCMC ...')
	results, _, _ = sampler.run_mcmc(burnin_params, iterations)
	vprint("Acceptance fraction of walkers:")
	vprint(sampler.acceptance_fraction)

	time_mcmc = timeit.default_timer() - itime_mcmc
	vprint("MCMC execution time: {:.4f} usec".format(time_mcmc))
	
	# Choosing a random chain from the emcee run
	samples = sampler.flatchain
	final_params = np.median(samples, axis=0)
	if verbose:
		vprint("Hyperparameters from MCMC:")
		print_params(final_params, priors)

	if plot_sample:
		# Plotting the results from the MCMC method
		fig1 = plt.figure(1, figsize=(14, 7), dpi=100)
		ax1 = fig1.add_subplot(111)
		
		# Plot initial data with errors in both subplots
		ax1.errorbar(time/(24*3600), flux, yerr=error, fmt=".k", capsize=0, label= 'Flux', markersize='3', elinewidth=1)
		x = np.linspace(min(time), max(time), num=1000)

		# Setup GP
		gp = setup_gp(final_params)
		#gp.compute(time/1e6, error)
		gp.compute(time/1e6)
		
		# Plot conditional predictive distribution of the model in upper plot
		mu, cov = gp.predict(flux, x/1e6)
		std = np.sqrt(np.diag(cov))
		std = np.nan_to_num(std)
		
		ax1.plot(x/(24*3600), mu, color="#ff7f0e", label= 'Mean distribution GP', linewidth=0.5)
		ax1.fill_between(x/(24*3600), mu+std, mu-std, color="#ff7f0e", alpha=0.6, edgecolor="none", label= '1 sigma', linewidth=0.6)
		#ax1.fill_between(x, mu+2*std, mu-2*std, color="#ff7f0e", alpha=0.3, edgecolor="none", label= '2 sigma', linewidth=0.5)
		#ax1.fill_between(x, mu+3*std, mu-3*std, color="#ff7f0e", alpha=0.15, edgecolor="none", label= '3 sigma', linewidth=0.5)
		
		ax1.set_xlabel('Time [days]',fontsize=15)	
		ax1.set_ylabel('Flux[ppm]',fontsize=15)
		ax1.tick_params(axis='both', which='major', labelsize=16)
		ax1.legend(loc='upper left', fontsize=15)

	if plot_corner:
		labels = [priors[i][0] for i in priors]
		fig2 = corner.corner(samples, labels=labels, 
							 quantiles=[0.5], show_titles=True, title_fmt='.3f',
							 truths=final_params, figsize=(14, 14), dpi=100, num=2)

	plot_psd = False
	if plot_psd:
		nyquist = (1 / (2*(time[1]-time[0])))*1e6
		f_sampling = 1 / (27.4*24*3600 / 1e6)
		freq = np.linspace(0.0, nyquist, (nyquist/f_sampling)+1 )

	return final_params


# Minimization Method
def run_minimization(data, priors, plot_sample=False, init_params=None, module='celerite'):

	# Timing execution time
	vprint('Starting Minimization ...')
	itime_min = timeit.default_timer()

	time, flux, error = data
	# Setting up initial parameters and runinng scipy minimize
	init_params = sample_priors(priors, 1)
	results = op.minimize(log_likelihood, init_params, args=data + (priors, gp), method='nelder-mead', tol=1e-18)

	time_min = timeit.default_timer() - itime_min
	vprint("Minimization execution time: {:.4f} usec\n".format(time_min))

	# Setting up GP using results from minimization
	final_params = results.x
	if verbose:
		vprint("Hyperparameters from minimization:")
		print_params(final_params, priors)
	
	if plot_sample:

		# Plotting the results from the minimization method
		fig = plt.figure(10, figsize=(14, 14), dpi=100)
		ax1 = fig.add_subplot(211)

		# Plot initial data with errors in both subplots
		ax1.errorbar(time/(24*3600), flux, yerr=error, fmt=".k", capsize=0, label= 'Flux', markersize='3', elinewidth=1)
		x = np.linspace(min(time), max(time), num=Nmax)

		# Setup GP
		gp = setup_gp(final_params)
		gp.compute(time/1e6, error)
		
		# Plot conditional predictive distribution of the model in upper plot
		mu, cov = gp.predict(flux, x/1e6)
		std = np.sqrt(np.diag(cov))
		std = np.nan_to_num(std)
		
		ax1.plot(x/(24*3600), mu, color="#ff7f0e", label= 'Mean distribution GP', linewidth=0.3)
		ax1.fill_between(x/(24*3600), mu+3*std, mu-3*std, color="#ff7f0e", alpha=0.15, edgecolor="none", label= '3 sigma', linewidth=0.5)
		#ax1.fill_between(x, mu+2*std, mu-2*std, color="#ff7f0e", alpha=0.3, edgecolor="none", label= '2 sigma', linewidth=0.5)
		#ax1.fill_between(x, mu+std, mu-std, color="#ff7f0e", alpha=0.5, edgecolor="none", label= '1 sigma', linewidth=0.5)

		ax1.title.set_text("Probability distribution for the gaussian process with the parameters from the minimization")
		ax1.set_xlabel('Time [days]',fontsize=15)	
		ax1.set_ylabel('Flux[ppm]',fontsize=15)
		ax1.tick_params(axis='both', which='major', labelsize=16)
		ax1.legend(loc='upper left', fontsize=15)

	return final_params