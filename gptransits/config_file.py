''' Filipe Pereira - 2017
Configuration file for the gp-tests methods
'''

import numpy as np

#Filenames or lists with stellar data
filename = 'sample19/KIC012008916.dat'
#filelist = 'list.dat'

# Filename where to write or append the results
results_file = 'test'

# Keywords for the fits file data
fits_options = {}
fits_options['time'] = 'TIME'
fits_options['flux'] = 'PDCSAP_FLUX'
fits_options['error'] = 'PDCSAP_FLUX_ERR'

#------------------------------------------------------------
# Settings for defining the priors for all parameters
prior_settings = {}
prior_settings[0] = [r'$A_1$', 'uniform', 30, 200]
prior_settings[1] = [r'$\omega_{0,1}$', 'uniform', 10, 90]

#prior_settings[2] = [r'$A_2$', 'uniform', 40, 200]
#prior_settings[3] = [r'$\omega_{0,2}$', 'uniform', 70, 200]

prior_settings[2] = [r'$A_{bump}$', 'uniform', 10, 250]
prior_settings[3] = [r'$Q_{bump}$', 'uniform', 1.2, 10]
prior_settings[4] = [r'$\omega_{bump}$', 'uniform', 100, 200]

#prior_settings[5] = ['Jitter', 'uniform', 0, 10]
#------------------------------------------------------------

# Other parameters
plot_sample = True
plot_corner = True
verbose = True

Nmax = 1317
offset = 0
#module = 'celerite'

nwalkers = 24
iterations = 2000