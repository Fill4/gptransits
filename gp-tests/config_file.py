#!/home/fill/anaconda3/bin/python
# File with data, fits format

#filename = 'kplr009267654_d21_v1.dat'
filename = None
listfile = 'list.dat'

results_file = 'george_expsq'

# Keywords for the fits file data
fits_options = {}
fits_options['time'] = 'TIME'
fits_options['flux'] = 'PDCSAP_FLUX'
fits_options['error'] = 'SAP_FLUX_ERR'

# Settings for defining the priors for all parameters
prior_settings = {}
prior_settings[0] = ['Amplitude', 'uniform', 0.0, 1.0]
prior_settings[1] = ['Timescale', 'uniform', 0.0, 1.0]
prior_settings[2] = ['Jitter', 'uniform', 0.0, 1.0]

# Other parameters
plot = True
verbose = True

Nmax = 1300
module = 'george'

nwalkers = 20
burnin = 400
iterations = 2000