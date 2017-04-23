# File with data, fits format
filename = 'Kepler91.fits'

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
Nmax = 500
plot = True