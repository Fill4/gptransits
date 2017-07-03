#!/home/fill/anaconda3/bin/python
from config_file import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import timeit
import sys, os, os.path
import gp_backend

if (filename):
	star_list = []
	star_list.append(filename)
else:
	star_list = []
	f = open(listfile)
	for file in f.readlines():
		if not (file.startswith('#')):
			star_list.append(file.rstrip())

# If there is no file , create it
if os.path.exists(results_file):
    pass
else:
	z = open(results_file + '.results', 'w')
	z.close()

for file in star_list:
	z = open(results_file + '.results', 'a')
	z.write(file + '\t')
	z.close()
	filename = 'RGBensemble/' + file

	# Start timer
	gp_backend.verboseprint('\n{:_^60}'.format('Starting GP fitting procedure'))
	startTimeScript = timeit.default_timer()

	# Bundle data in tuple for organisation
	if filename.endswith('.fits'):
		data = gp_backend.readfits(filename, Nmax)
	else:
		data = gp_backend.readtxt(filename, Nmax)
	#data = (time, flux, error)

	# Initiate prior distributions according to options set by user
	priors = gp_backend.setup_priors(prior_settings)

	# Run minimization
	gp_backend.run_minimization(data, priors, plot=plot, module=module)

	# Run MCMC
	gp_backend.run_mcmc(data, priors, plot=plot, nwalkers=nwalkers, burnin=burnin, iterations=iterations, module=module)

	if plot:
		for i in plt.get_fignums():
			plt.figure(i)
			plt.savefig('Figures/%s%s%d.png' % (os.path.splitext(os.path.basename(filename))[0], '_fig', i), dpi = 200)
			plt.clf()
			plt.cla()
			plt.close()


	# Print execution time
	fullTimeScript = timeit.default_timer() - startTimeScript
	gp_backend.verboseprint("\nComplete execution time: {:10.5} usec".format(fullTimeScript))
	gp_backend.verboseprint('\n{:_^60}\n'.format('END'))
