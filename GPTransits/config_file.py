''' Filipe Pereira - 2017
Configuration file for the gp-tests methods
'''

#Filenames or lists with stellar data
filename = 'kplr009267654_d21_v1.dat'
filelist = 'list.dat'
# Filename where to write or append the results
results_file = 'celerite_test'

# Keywords for the fits file data
fits_options = {}
fits_options['time'] = 'TIME'
fits_options['flux'] = 'PDCSAP_FLUX'
fits_options['error'] = 'SAP_FLUX_ERR'

# Settings for defining the priors for all parameters
prior_settings = {}
prior_settings[0] = ['Amplitude', 'uniform', 0.0, 1.0]
prior_settings[1] = ['Timescale', 'uniform', 0.0, 100.0]
prior_settings[2] = ['Jitter', 'uniform', 0.0, 0.1]

# Other parameters
plot = False
verbose = True

Nmax = 500
module = 'celerite'

nwalkers = 20
burnin = 400
iterations = 2000