''' Filipe Pereira - 2017
Configuration file for the gp-tests methods
'''

#Filenames or lists with stellar data
#filename = 'kplr009267654_d21_v1.dat'
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
prior_settings[0] = ['Amplitude_1', 'uniform', 0.0, 0.1]
prior_settings[1] = ['Timescale_1', 'uniform', 10.0, 300.0]
#prior_settings[2] = ['Jitter', 'uniform', 0.0, 0.1]
prior_settings[2] = ['Amplitude_2', 'uniform', 0.0, 0.1]
prior_settings[3] = ['Timescale_2', 'uniform', 10.0, 300.0]

# Other parameters
plot = True
verbose = True

Nmax = 1300
module = 'celerite'

nwalkers = 20
burnin = 1000
iterations = 5000