#!/usr/bin/env python3

''' Filipe Pereira - 2017
General procedure for fitting light curve data to a model and using
gaussian processes to fit the remaining stochastic noise
'''

import gptransits
import logging
from components_v2 import *

# INPUT DATA ---------------------------------------------
# Keywords for the fits file data
fits_options = {}
fits_options['time'] = 'TIME'
fits_options['flux'] = 'PDCSAP_FLUX'
fits_options['error'] = 'PDCSAP_FLUX_ERR'

#------------------------------------------------------------
# Settings for defining the priors for all parameters
prior_settings = {}
prior_settings[0] = [r'$A_{bump}$', 'uniform', 10, 250]
prior_settings[1] = [r'$Q_{bump}$', 'uniform', 1.2, 10]
prior_settings[2] = [r'$\omega_{bump}$', 'uniform', 100, 200]

prior_settings[3] = [r'$A_1$', 'uniform', 30, 200]
prior_settings[4] = [r'$\omega_{0,1}$', 'uniform', 10, 90]

prior_settings[5] = [r'$A_2$', 'uniform', 40, 200]
prior_settings[6] = [r'$\omega_{0,2}$', 'uniform', 70, 200]

prior_settings[7] = ['Jitter', 'uniform', 0, 100]
#------------------------------------------------------------

model = Model(OscillationBump(), Granulation(), Granulation(), WhiteNoise())

# Other parameters
plot_flags = {'plot_gp':0, 'plot_corner':0, 'plot_psd':0}

verbose = True
if verbose:
	logging.basicConfig(format='%(message)s', level=logging.INFO)

nwalkers = 24
iterations = 2000
burnin = 500

#--------------------------------------------------------------
#--------------------------------------------------------------


data_dir = './results/TEST/'
gptransits.main(data_dir, data_dir, model, prior_settings, plot_flags, nwalkers, iterations, burnin)