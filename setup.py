from setuptools import setup, find_packages
setup(
    name='gptransits',
    version='0.1',
    author='Filipe Pereira',
    author_email='filipe.pereira@astro.up.pt',
    license='MIT',

    packages=['gptransits'],
    install_requires=['emcee', 'george', 'celerite>=0.2.0', 'corner', 'pyfits'],
    description='Fit time series to light curves with correlated noise modelled by gaussian processes',
    
)