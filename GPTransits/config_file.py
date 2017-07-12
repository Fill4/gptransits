''' Filipe Pereira - 2017
Configuration file for the gp-tests methods
'''

#Filenames or lists with stellar data
#filename = 'kplr007060732_d21_v1.dat'
filename = 'kplr008366239_mast.fits'
filelist = 'kiclist.dat'
# Filename where to write or append the results
results_file = 'temp_celerite'

# Keywords for the fits file data
fits_options = {}
fits_options['time'] = 'TIME'
fits_options['flux'] = 'PDCSAP_FLUX'
fits_options['error'] = 'PDCSAP_FLUX_ERR'

# Settings for defining the priors for all parameters
prior_settings = {}
prior_settings[0] = [r'$A_1$', 'uniform', 0.0, 2000.0]
prior_settings[1] = [r'$\omega_{0,1}$', 'uniform', 0.0, 100.0]
#prior_settings[2] = ['Amplitude_2', 'uniform', 0.0, 6000.0]
#prior_settings[3] = ['Timescale_2', 'uniform', 4.0, 30.0]
#prior_settings[4] = ['Amplitude_3', 'uniform', 0.0, 50000.0]
#prior_settings[5] = ['Timescale_3', 'uniform', 0.0, 0.5]
#prior_settings[6] = ['Amplitude_4', 'uniform', 0.0, 0200.0]
#prior_settings[7] = ['Timescale_4', 'uniform', 0.0, 120.0]
#prior_settings[4] = ['Jitter', 'uniform', 0.0, 100.0]

# Other parameters
plot = True
verbose = True

Nmax = 1300
module = 'celerite'

nwalkers = 30
burnin = 800
iterations = 2500