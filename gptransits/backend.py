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

if verbose:
	def verboseprint(*args):
		# Print each argument separately so caller doesn't need to
        # stuff everything to be printed into a single string
		for arg in args:
			print(arg,)
		print()
else:   
	verboseprint = lambda *a: None # do-nothing function

def readfits(filename, Nmax, offset, fits_options):
	# Read data from fits and remove nans
	hdulist = pyfits.open(filename)

	ntime = getattr(hdulist[1].data, fits_options['time'])[offset:Nmax+offset]
	nflux = getattr(hdulist[1].data, fits_options['flux'])[offset:Nmax+offset]
	nerror = getattr(hdulist[1].data, fits_options['error'])[offset:Nmax+offset]
	#nerror[~np.isfinite(nerror)] = 10

	ind = np.logical_and(~np.isnan(ntime), ~np.isnan(nflux), ~np.isnan(nerror))
	time = ntime[ind]
	time = time - time[0]
	flux = nflux[ind]/np.median(nflux[ind])
	error = nerror[ind]/np.median(nflux[ind])
	flux = (flux - 1)*1e6 #to ppm
	error = error*1e6 #to ppm
	#flux = nflux[ind]

	#print(len(getattr(hdulist[1].data, fits_options['time'])))
	#plt.errorbar(time, flux, yerr=error, fmt=".k", capsize=0)
	#plt.show()
	#sys.exit()

	return (time, flux, error)

def readtxt(filename, Nmax, offset):
	buffer = np.loadtxt(filename)

	#ntime = buffer[offset:Nmax+offse,0]
	#nflux = buffer[offset:Nmax+offse,1]
	#nerror = np.ones(len(nflux)) * 1e-5
	
	ntime = buffer[:,0]
	nflux = buffer[:,1]
	nerror = buffer[:,2]

	time = ntime
	flux = nflux
	error = nerror

	"""
	ind = np.logical_and(~np.isnan(ntime), ~np.isnan(nflux), ~np.isnan(nerror))
	time = ntime[ind]
	time = time - time[0]
	flux = nflux[ind]/np.median(nflux[ind])
	error = nerror[ind]/np.median(nflux[ind])
	flux = (flux - 1)*1e6
	error = error*1e6
	"""

	#plt.errorbar(time, flux, yerr=error, fmt=".k", capsize=0)
	#plt.show()
	#sys.exit()

	return (time, flux, error)

# Calls the correct gp module according to user definition
def setup_gp(pars, module='george'):
	if module == 'george':
		gp = setup_george(pars)
	elif module == 'celerite':
		gp = setup_celerite(pars)
	else:
		sys.exit('Choose valid gaussian process module')
	return gp
	
# Returns a GP object from the george module with the provided parameters pars
def setup_george(pars):
	#t1, t2, jitter = pars
	t1, t2 = pars
	k1 = t1**2 * george.kernels.ExpSquaredKernel(t2**2)
	#k2 = george.kernels.WhiteKernel(jitter**2)
	kernel = k1 #+ k2
	gp = george.GP(kernel)
	return gp

# Returns a GP object from the celerite module with the provided parameters pars
def setup_celerite(pars):
	#S1, w1, S2, w2, S3, w3, S4, w4, jitter = pars
	#S1, w1, S2, w2, S3, w3, S4, w4 = pars
	#S1, w1, S2, w2, S3, w3, jitter = pars
	#S1, w1, S2, w2, S3, w3 = pars
	#S1, w1, S2, w2, jitter = pars
	#S1, w1, S2, w2 = pars
	#S1, w1, jitter = pars
	#S1, w1 = pars
	S1, w1, S2, w2, S_bump, w_bump, Q_bump = pars

	Q = 1.0 / np.sqrt(2.0)
	terms = []
	
	'''
	for i in range(int(np.floor(len(pars)/2))):
		kernel_temp = celerite.terms.SHOTerm(log_S0=np.log(pars[i*2]), log_Q=np.log(Q) , log_omega0=np.log(i*2+1))
		kernel_temp.freeze_parameter("log_Q")
		terms.append(kernel_temp)
	if len(pars)%2 == 1:
		terms.append(celerite.terms.JitterTerm(log_sigma=np.log(pars[-1])))
	kernel = terms[0]+terms[1]+terms[2]+terms[3]
	'''
	
	#'''
	kernel_1 = celerite.terms.SHOTerm(log_S0=np.log(S1), log_Q=np.log(Q), log_omega0=np.log(w1))
	kernel_1.freeze_parameter("log_Q")
	kernel_2 = celerite.terms.SHOTerm(log_S0=np.log(S2), log_Q=np.log(Q), log_omega0=np.log(w2))
	kernel_2.freeze_parameter("log_Q")
	#kernel_3 = celerite.terms.SHOTerm(log_S0=np.log(S3), log_Q=np.log(Q), log_omega0=np.log(w3))
	#kernel_3.freeze_parameter("log_Q")
	#kernel_4 = celerite.terms.SHOTerm(log_S0=np.log(S3), log_Q=np.log(Q), log_omega0=np.log(w3))
	#kernel_4.freeze_parameter("log_Q")


	kernel = kernel_1 + kernel_2 #+ kernel_3 #+ kernel_4
	kernel += celerite.terms.SHOTerm(log_S0=np.log(S_bump), log_Q=np.log(Q_bump), log_omega0=np.log(w_bump))
	#kernel += celerite.terms.JitterTerm(log_sigma=np.log(jitter))
	#'''
	
	#print(terms)
	#print(kernel.get_parameter_dict())
	gp = celerite.GP(kernel)
	return gp

def print_pars(pars, priors):
	for par in priors:
		verboseprint("{:12} {:1} {:2} {:16.12f}".format(priors[par][0], "-", "", pars[par]))

# Defining functions to handle the Monte Carlo method calculations
# Calculates the value of the model for parameters pars in the positions phase
def model(pars, phase):
	return np.zeros(len(phase))

# Creates the dictionary priors with the instantiated probability distributions for each parameter 
def setup_priors(prior_settings):
	priors = {}
	for par in prior_settings:
		priors[par] = [prior_settings[par][0], getattr(scipy.stats, prior_settings[par][1])(loc=prior_settings[par][2], scale=prior_settings[par][3])]
	return priors

# Draws num number of samples from each of the parameters distribution
def sample_priors(priors, num = 1):
	sample = np.zeros([len(priors), num])
	for par in priors:
		sample[par] = priors[par][1].rvs(num)
	return sample

# Defines the priors for all the parameters
def log_prior(pars, priors):
	prior_sum = 0
	# Evaluate parameters according to priors. Return sum when working in log
	for par in priors:
		prior_sum += priors[par][1].logpdf(pars[par])
	return prior_sum

# Calculates the lnlikelihood of the parameters pars proposed. Can include model and
# prior calculation besides the gaussian process
def log_likelihood(pars, phase, flux, error, priors, minimization=False, module='george'):
	gp = setup_gp(pars, module)
	gp.compute(phase, error)
	if module == 'george':
		lnlikelihood = gp.lnlikelihood(flux - model(pars, phase), quiet=True)
	elif module == 'celerite':
		lnlikelihood = gp.log_likelihood(flux - model(pars, phase))
	else:
		sys.exit('Choose valid gaussian process module')

	if (minimization): # scipy minimize has problems with infinities
		return -lnlikelihood if np.isfinite(lnlikelihood) else 1e25
	else:
		prior = log_prior(pars, priors)
		return prior + lnlikelihood if np.isfinite(prior) else -np.inf

# Calculates the gradient of the lnlikelihood for the proposed parameters pars
# This quantity is used in minimization methods such as scipy minimize
# Commented because it's unnecessary to the minimization right now
''' 
def lnLikeGrad(pars, phase, flux, error, minimization=False):
	gp = setup_george(pars)
	gp.compute(phase, error)
	return -gp.grad_lnlikelihood(flux, quiet=True)
'''

# Minimization Method
def run_minimization(data, priors, plot_sample=False, init_pars=None, module='celerite'):

	# Timing execution time
	verboseprint('\n{:_^60}\n'.format('Starting Minimization'))
	startTimeMinimization = timeit.default_timer()

	# Setting up initial parameters and runinng scipy minimize
	init_pars = sample_priors(priors, 1)
	results = op.minimize(log_likelihood, init_pars, args=data + (priors, True, module), method='nelder-mead', tol=1e-18)

	fullTimeMinimization = timeit.default_timer() - startTimeMinimization
	verboseprint("Minimization execution time: {:10.5} usec\n".format(fullTimeMinimization))

	phase, flux, error = data
	# Setting up GP using results from minimization
	final_pars = results.x
	gp = setup_gp(final_pars, module)
	gp.compute(phase, error)
	# Printing hyperparameters
	verboseprint("Hyperparameters from minimization:")
	print_pars(final_pars, priors)

	if plot_sample:

		# Plotting the results from the minimization method
		fig = plt.figure(1, figsize=(14, 14), dpi=100)
		ax1 = fig.add_subplot(211)
		ax2 = fig.add_subplot(212)
		# Plot initial data with errors in both subplots
		ax1.errorbar(phase, flux, yerr=error, fmt=".k", capsize=0, label= 'Flux', markersize='3', elinewidth=1)
		ax2.errorbar(phase, flux, yerr=error, fmt=".k", capsize=0, label= 'Flux', markersize='3', elinewidth=1)
		numpoints = (max(phase)-min(phase))/(1.0/48.0)
		x = np.linspace(min(phase), max(phase), numpoints)
		# Plot conditional predictive distribution of the model in upper plot
		mu, cov = gp.predict(flux, x)
		std = np.sqrt(np.diag(cov))
		std = np.nan_to_num(std)
		ax1.plot(x, mu, color="#ff7f0e", label= 'Mean distribution GP', linewidth=0.3)
		ax1.fill_between(x, mu+3*std, mu-3*std, color="#ff7f0e", alpha=0.15, edgecolor="none", label= '3 sigma', linewidth=0.5)
		ax1.fill_between(x, mu+2*std, mu-2*std, color="#ff7f0e", alpha=0.3, edgecolor="none", label= '2 sigma', linewidth=0.5)
		ax1.fill_between(x, mu+std, mu-std, color="#ff7f0e", alpha=0.5, edgecolor="none", label= '1 sigma', linewidth=0.5)
		ax1.title.set_text("Probability distribution for the gaussian process with the parameters from the minimization")
		ax1.set_ylabel('Flux')
		ax1.legend(loc='upper left')
		# Plot sample from the conditional ditribution in lower plot
		#m = gp.sample_conditional(flux, x)
		#ax2.plot(x, m, color="#4682b4", label='Sample')
		ax2.plot(x, np.random.normal(loc=mu, scale=std), color="#4682b4", label='Sample')
		ax2.title.set_text("Sample drawn from the probability distribution above")
		ax2.set_xlabel('Time')	
		ax2.set_ylabel('Flux')
		ax2.legend(loc='upper left')

	return final_pars, fullTimeMinimization

def run_mcmc(data, priors, plot_corner=True, plot_sample=False, init_pars=None, nwalkers=20, burnin=500, iterations=2000, module='celerite'):
	
	verboseprint('\n{:_^60}\n'.format('Starting MCMC'))
	startTimeMCMC = timeit.default_timer()

	# Draw samples from the prior distributions to have initial values for all the walkers
	init_pars = sample_priors(priors, nwalkers)
	ndim = len(init_pars)

	# Initiate the sampler
	sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=data + (priors, False, module))

	#Burn-in
	burnin_pars, _, _ = sampler.run_mcmc(init_pars.T, burnin)
	verboseprint("Acceptance fraction of burn-in walkers:")
	verboseprint(sampler.acceptance_fraction)
	
	sampler.reset()

	#Run the walkers
	results, _, _ = sampler.run_mcmc(burnin_pars, iterations)
	verboseprint("Acceptance fraction of final walkers:")
	verboseprint(sampler.acceptance_fraction)

	fullTimeMCMC = timeit.default_timer() - startTimeMCMC
	verboseprint("\nMCMC execution time: {:10.5} usec\n".format(fullTimeMCMC))
	
	phase, flux, error = data
	# Choosing a random chain from the emcee run
	samples = sampler.flatchain
	final_pars = np.median(samples, axis=0)
	verboseprint("Hyperparameters from MCMC:")
	print_pars(final_pars, priors)

	# Testing forced result
	#final_pars = (175.0, 12.0)

	# Set up the GP for this sample.
	gp = setup_gp(final_pars, module)
	gp.compute(phase, error)

	if plot_sample:

		# Plotting the results from the MCMC method
		fig1 = plt.figure(2, figsize=(14, 7), dpi=100)
		ax1 = fig1.add_subplot(111)
		#ax2 = fig1.add_subplot(212)
		
		# Plot initial data with errors in both subplots
		ax1.errorbar(phase, flux, yerr=error, fmt=".k", capsize=0, label= 'Flux', markersize='3', elinewidth=1)
		numpoints = (max(phase)-min(phase))/(1.0/48.0)
		x = np.linspace(min(phase), max(phase), num=numpoints)
		
		# Plot conditional predictive distribution of the model in upper plot
		mu, cov = gp.predict(flux, x)
		std = np.sqrt(np.diag(cov))
		std = np.nan_to_num(std)
		
		ax1.plot(x, mu, color="#ff7f0e", label= 'Mean distribution GP', linewidth=0.5)
		ax1.fill_between(x, mu+std, mu-std, color="#ff7f0e", alpha=0.6, edgecolor="none", label= '1 sigma', linewidth=0.6)
		#ax1.fill_between(x, mu+2*std, mu-2*std, color="#ff7f0e", alpha=0.3, edgecolor="none", label= '2 sigma', linewidth=0.5)
		#ax1.fill_between(x, mu+3*std, mu-3*std, color="#ff7f0e", alpha=0.15, edgecolor="none", label= '3 sigma', linewidth=0.5)
		
		#ax1.set_title("Probability distribution for the gaussian process with the parameters from the MCMC",fontsize=16)
		ax1.set_xlabel('Time [days]',fontsize=15)	
		ax1.set_ylabel('Flux[ppm]',fontsize=15)
		ax1.tick_params(axis='both', which='major', labelsize=16)
		ax1.legend(loc='upper left', fontsize=15)
		
		# Compute the prediction conditioned on the observations and plot it.
		# Plot sample from the conditional ditribution in lower plot
		#m = gp.sample_conditional(flux, x)
		#ax2.plot(x, m, color="#4682b4", label= 'Sample')
		#ax2.plot(x, np.random.normal(loc=mu, scale=std), color="#4682b4", label='Sample')
		#ax2.set_title("Sample drawn from the probability distribution above",fontsize=16)
		#ax2.set_ylabel('Flux [ppm]',fontsize=18)
		#ax2.legend(loc='upper left')

	if plot_corner:
		labels = [priors[i][0] for i in priors]
		fig2 = corner.corner(samples, labels=labels, 
							 quantiles=[0.5], show_titles=True, title_fmt='.3f',
							 truths=final_pars, figsize=(14, 14), dpi=100, num=3)

	return final_pars