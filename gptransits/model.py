#!/usr/bin/env python3
"""
Fits a transit model together with a Gaussian Process model created by celerite to lightcurve observations
"""

import sys, os
import pickle
import time
import logging
from pathlib import Path
import importlib
from multiprocessing.pool import Pool
import signal

import numpy as np
import matplotlib.pyplot as plt
import emcee
import celerite
import lightkurve as lk

# Model imports
from .transit import BatmanModel#, PysyzygyModel
from .gp import GPModel
from .settings import Settings

# Analysis imports
from .convergence import geweke, gelman_rubin, gelman_brooks
from .stats import mapv, mode, hpd
from .plot import *

__all__ = ["Model"]

log = logging.getLogger("my_logger")

class Model():
    # Directories, configuration dict and settings struct # TODO: Move to general struct possibly
    workdir = None
    outdir = None
    figdir = None
    cfg = None
    settings = None

    # LightCurve object from lightkurve
    lc = None

    # Static object handles for the models. For multiprocessing
    mean_model = None
    gp_model = None
    gp = None
    # Static handle to Likelihood function defined according to models defined
    lnlike_func = None

    # MCMC static settings
    mean_model_ndim = 0
    gp_model_ndim = 0
    ndim = 0

    # Sampler data
    backend = None
    chain = None
    log_prob = None

    # Analysis
    stats = None
    stats_flag = False

    @classmethod
    def __init__(cls, lc, cfg=None, workdir=None, quiet=False, logfile=None):
        # Setp the main logger. Clear the logger handlers list as this is a static class and we would accumulate
        log.handlers = []
        if quiet:
            loglevel = logging.WARN
        else:
            loglevel = logging.INFO
        log.setLevel(loglevel)
        if logfile is None:
            sh = logging.StreamHandler()
            sh.setLevel(loglevel)
            formatter = logging.Formatter('[%(levelname)s] %(message)s')
            sh.setFormatter(formatter)
            log.addHandler(sh)
        else:
            fh = logging.FileHandler("log.txt", filemode="w")
            fh.setLevel(loglevel)
            formatter = logging.Formatter('[%(levelname)s] %(message)s')
            fh.setFormatter(formatter)
            log.addHandler(fh)

        # Setup workdir workdir
        if workdir is None:
            cls.workdir = Path(".")
        else:
            cls.workdir = Path(workdir)
        log.info(f'Working in directory: {cls.workdir.resolve()}')
        cls.outdir = cls.workdir / "output"
        cls.figdir = cls.workdir / "figures"

        # Define config file path and try to get config from file
        if cfg is None:
            cfg_file = cls.workdir / "config.py"
        else:
            cfg_file = Path(cfg)
        try:
            spec = importlib.util.spec_from_file_location("config", cfg_file)
            cls.cfg = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cls.cfg)
        except Exception:
            log.exception(f"Couldn't import {cfg_file.stem}")
            sys.exit()
        sys.modules["cfg"] = cls.cfg

        # Define settings either from config file or default
        if hasattr(cls.cfg, "settings"):
            cls.settings = cls.cfg.settings
        else:
            cls.settings = Settings()

        # Set random state early to control the initial random sample and also emcee iterations
        np.random.seed(cls.settings.seed)

        # If the lc passed was a string, read it into a lk.LightCurve object
        if isinstance(lc, str):
            cls.load_lc_from_file(Path(lc))
        elif isinstance(lc, lk.LightCurve):
            cls.lc = lc

        # Setup the gp and transit models from the configuration files
        cls.setup_models()
        cls.load_data()
        cls.stats_flag = False # Not sure how the static class handles multiple creation

    @classmethod
    def load_lc_from_file(cls, lc_file):
        # Read data from file
        log.info(f'Reading lightcurve from: {lc_file.name}')
        try:
            time, flux, flux_err = np.loadtxt(lc_file, unpack=True)
        except ValueError:
            if cls.settings.include_errors:
                log.error(f"Need a flux_err column when include_errors is True")
                sys.exit()
            try:
                time, flux = np.loadtxt(lc_file, unpack=True)
                flux_err = None
            except ValueError:
                log.error(f"Couldn't load data from the lc_file: {lc_file}")
                sys.exit()
                

        # TODO: Needs to correct for the format of data being processed
        cls.lc = lk.LightCurve(time = np.copy(time), flux = np.copy(flux) * 1e6, flux_err = flux_err)
        cls.lc.meta["lc_file"] = lc_file

    @classmethod
    def setup_models(cls):

        # Setup a transit model if there is config in file. # TODO: Add other models
        if hasattr(cls.cfg, "transit"):
            cls.mean_model = BatmanModel(cls.cfg.transit["name"], cls.cfg.transit["params"])
            cls.mean_model.init_model(cls.lc.time, cls.lc.time[1]-cls.lc.time[0], 6)
            cls.mean_model_ndim = cls.mean_model.mask.sum()

        # Setup gp model according to the config file
        if hasattr(cls.cfg, "gp"):
            cls.gp_model = GPModel(cls.cfg.gp)
            kernel = cls.gp_model.get_kernel(cls.gp_model.sample_prior()[0])
            cls.gp_model_ndim = kernel.get_parameter_vector().size

            # Setup celerite gp with kernel. Time goes in microseconds for the muHz parameters in the GP
            cls.gp = celerite.GP(kernel)
            days_to_microsec = (24*3600) / 1e6
            if cls.settings.include_errors:
                cls.gp.compute(cls.lc.time*days_to_microsec, yerr=cls.lc.flux_err)
            else:
                cls.gp.compute(cls.lc.time*days_to_microsec)

        # Define the log_likelihood (cost) function according to the data we have. Works as a function pointer
        if cls.mean_model is not None and cls.gp_model is not None:
            cls.lnlike_func = cls.full_lnlike
            log.info("Model with GP and transit")
        elif cls.gp_model is not None:
            cls.lnlike_func = cls.gp_lnlike
            log.info("Model with GP")
        elif cls.mean_model is not None:
            cls.lnlike_func = cls.transit_lnlike
            log.info("Model with transit")
        else:
            log.error("Need to define at least a gp or transit model")
            sys.exit()
    
    @classmethod
    def load_data(cls):
        # Define backend if we want to save progress and load the data there
        if cls.settings.save:
            if not cls.outdir.is_dir():
                cls.outdir.mkdir()
            filename = cls.outdir / "chain.hdf5"
            cls.backend = emcee.backends.HDFBackend(str(filename), name="gptransits")

            # TODO: Probably doesnt work right now because of chain dimensions return in backend
            # TODO: Update to emcee latest version and account for this change
            # if filename.is_dir():
            #     cls.chain = cls.backend.get_chain()
            #     cls.log_prob = cls.backend.get_log_prob()

        # Otherwise set backend to None
        else:
            cls.backend = None

    @classmethod
    def init_worker(cls):
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    @classmethod
    def run(cls, reset=False, num_threads=1):
        # -------------------- MCMC ----------------------------
        cls.ndim = cls.mean_model_ndim + cls.gp_model_ndim
        nwalkers =  cls.ndim * 4
        nsteps = cls.settings.num_steps
        # By default we define the initial params from the priors
        set_params = True
        # Check backend status and iterations and compare to config. Reset if flag is true
        if cls.backend is not None:
            # If we have a backend file to fetch the sampler from
            filename = Path(cls.backend.filename)
            if filename.is_file():
                # If we want to reset it, just clear the sampler and get init sample from prior
                if reset:
                    log.info("Resetting the backend sampler")
                    # backend.reset(nwalkers, ndim) # TODO: This line is not working as intended
                    filename.unlink() # TODO: Remove the file. Hack because reset is not working
                # Else, init the backend and check if we have more iterations in the sampler than the desired ones
                else:
                    # If we have, stop the code
                    if cls.backend.iteration >= nsteps:
                        log.warn("Skipping run. Backend number of iterations greater than settings")
                        return
                    # Otherwise, calculate the remaining steps and continue from there
                    else:
                        nsteps = nsteps - cls.backend.iteration
                        set_params = False
            
        # If the initial params are not taken from the backend init them from the prior
        init_params = None
        if set_params:
            # Sample priors to get initial values for all walkers
            if cls.gp_model is not None and cls.mean_model is not None:
                init_gp_params = cls.gp_model.sample_prior(num=nwalkers)
                init_mean_params = cls.mean_model.sample_prior(num=nwalkers)
                init_params = np.hstack([init_gp_params, init_mean_params])
            elif cls.mean_model is not None:
                init_params = cls.mean_model.sample_prior(num=nwalkers)
            elif cls.gp_model is not None:
                init_params = cls.gp_model.sample_prior(num=nwalkers)

        # Single or Multiprocessing always uses pool as the init_worker can handle the system interrupts
        pool = Pool(num_threads, cls.init_worker)
        sampler = emcee.EnsembleSampler(nwalkers, cls.ndim, cls.lnlike_func, pool=pool, backend=cls.backend)

        # Run mcmc
        log.info(f"Running MCMC on {num_threads} processes for {nsteps} iterations")
        try:
            sampler.run_mcmc(init_params, nsteps, progress=True)
        except KeyboardInterrupt:
            log.warn(f"Emcee was stopped by user input")
            pool.terminate()
            pool.join()
            sys.exit()

        # TODO: get_chain is the new method but it has different dims
        cls.chain = sampler.chain.copy() 
        cls.log_prob = sampler.get_log_prob().copy()

        # Save data # TODO: Need to update lnprobability name and then update filenames
        if cls.settings.save:
            log.info(f"Saving chain and log_prob")
            with open(cls.outdir / "chain.pk", "wb") as f:
                pickle.dump(sampler.chain, f, protocol=-1)
            with open(cls.outdir / "lnprobability.pk", "wb") as f:
                pickle.dump(sampler.get_log_prob(), f, protocol=-1)

    # -------------------------------------------------------------------------------------------
    # LIKELIHOOD FUNCTIONS
    # -------------------------------------------------------------------------------------------
    # Defines the likelihood function for emcee. Classmethod (global) to allow for multithreading
    @classmethod
    def full_lnlike(cls, params):
        gp_lnprior = cls.gp_model.lnprior(params[:cls.gp_model_ndim])
        mean_lnprior = cls.mean_model.lnprior(params[cls.gp_model_ndim:])
        if not (np.isfinite(gp_lnprior) and np.isfinite(mean_lnprior)):
            return -np.inf
        
        # Convert params to celerite model and update gp object
        gp_params = cls.gp_model.get_parameters_celerite(params[:cls.gp_model_ndim])
        cls.gp.set_parameter_vector(gp_params)

        mean = cls.mean_model.get_value(params[cls.gp_model_ndim:], cls.lc.time)
        lnlikelihood = cls.gp.log_likelihood(cls.lc.flux - mean)
            
        return mean_lnprior + gp_lnprior + lnlikelihood

    @classmethod
    def gp_lnlike(cls, params):
        gp_lnprior = cls.gp_model.lnprior(params)
        if not np.isfinite(gp_lnprior):
            return -np.inf
        
        # Convert params to celerite model and update gp object
        cls.gp.set_parameter_vector(cls.gp_model.get_parameters_celerite(params))

        lnlikelihood = cls.gp.log_likelihood(cls.lc.flux)
            
        return gp_lnprior + lnlikelihood

    @classmethod
    def transit_lnlike(cls, params):
        mean_lnprior = -cls.mean_model.lnprior(params[cls.gp_model_ndim:])
        if not np.isfinite(mean_lnprior):
            return -np.inf
    
        residuals = cls.lc.flux - cls.mean_model.get_value(params, cls.lc.time)
        # TODO: Needs white noise component added here maybe
        if cls.lc.flux_err is not None:
            lnlikelihood = np.sum(-0.5 * (residuals**2 / cls.lc.flux_err**2))
        else:
            lnlikelihood = np.sum(-0.5 * residuals**2)
        
        # if lnlikelihood > -18.5:
        # log.info(f"lnlikelihood: {lnlikelihood}")

        return mean_lnprior + lnlikelihood


    # -------------------------------------------------------------------------------------------
    # ANALYSIS
    # -------------------------------------------------------------------------------------------
    @classmethod
    def statistics(cls, fout=None):
        # Setup folder names and load chain and log_prob from the pickle files
        log.info(f"Running full analysis on: {cls.lc.meta['lc_file'].stem}")
        
        # Read in all data
        if cls.chain is not None and cls.log_prob is not None:
            log.info(f"Using data from current model")
            chain = cls.chain.copy()
            log_prob = cls.log_prob.copy()
        else:
            log.info(f"Fetching data from output directory: {cls.outdir}")
            if not cls.outdir.is_dir():
                log.error(f"No directory with target data to analyse: {cls.outdir}")
                sys.exit()

            log.info(f"Fetching: {cls.outdir}/chain.pk")
            with open(f"{cls.outdir}/chain.pk", "rb") as f:
                chain = pickle.load(f)
            log.info(f"Fetching: {cls.outdir}/lnprobability.pk")
            with open(f"{cls.outdir}/lnprobability.pk", "rb") as p:
                log_prob = pickle.load(p)

        if cls.gp_model is not None and cls.mean_model is not None:
            gp_names = cls.gp_model.get_parameters_latex()
            transit_names = cls.mean_model.get_parameters_latex()
            names = np.hstack([gp_names, transit_names])
        elif cls.mean_model is not None:
            names = cls.mean_model.get_parameters_latex()
        elif cls.gp_model is not None:
            names = cls.gp_model.get_parameters_latex()
        
        output_dict = {"name": cls.lc.meta["lc_file"].stem}

        # Run burn-in diagnostics (Geweke)
        log.info(f"Calculating Geweke diagnostic")
        starts, zscores = geweke(chain)
        chains_over = np.max(zscores, axis=2) > 2
        num_chains_over = np.sum(chains_over, axis=1)
        start_index = np.where(num_chains_over == np.min(num_chains_over))[0][0]
        if start_index == 0:
            start_index = 1
        
        walkers_mask = ~chains_over[start_index]
        geweke_flag = False
        if np.sum(chains_over[start_index])/chain.shape[0] > 0.4:
            walkers_mask[:] = True
            geweke_flag = True
        num_walkers = np.sum(walkers_mask)

        reduced_chain = chain[walkers_mask,starts[start_index]:,:]
        reduced_log_prob = log_prob[starts[start_index]:,walkers_mask]
        
        output_dict["geweke_flag"] = geweke_flag
        output_dict["num_walkers"] = num_walkers
        output_dict["percentage_walkers"] = starts[start_index]/chain.shape[1]

        # Cut chains with lower log_prob
        log.info(f"Removing lower log_prob chains")
        log_prob_mask = (np.median(reduced_log_prob, axis=0) - np.median(reduced_log_prob)) > -np.std(reduced_log_prob)

        reduced_chain = reduced_chain[log_prob_mask,:,:]
        reduced_log_prob = reduced_log_prob[:,log_prob_mask]

        # Check consistency between chains
        log.info(f"Calculating Gelman-Rubin diagnostic")
        r_hat, _, _ = gelman_rubin(reduced_chain)
        r_mvar = gelman_brooks(reduced_chain)

        log.info(f"Calculating Gelman-Brooks diagnostic")
        r_hat_str = "".join([" {:>9.5f}".format(ri) for ri in r_hat])
        
        output_dict["multi_rhat"] = r_mvar.max()
        output_dict["rhat"] = r_hat

        log.info(f"Calculating parameter statistics")
        params = {"names": names}
        samples = reduced_chain.reshape([reduced_chain.shape[0]*reduced_chain.shape[1], reduced_chain.shape[2]])
        params["median"] = np.median(samples, axis=0)
        params["hpd_down"], params["hpd_up"] = hpd(reduced_chain, level=0.683)
        hpd_99_down, hpd_99_up = hpd(reduced_chain, level=0.99)
        params["hpd_99_interval"] = (hpd_99_up - hpd_99_down)
        params["mapv"] = mapv(reduced_chain, reduced_log_prob)
        params["modes"] = mode(chain)

        # Get the median and 68% intervals for each of the parameters
        # log.info(f"Calculating medians and stds")
        # samples = reduced_chain.reshape([reduced_chain.shape[0]*reduced_chain.shape[1], reduced_chain.shape[2]])
        # percentiles = np.percentile(samples.T, [50,16,84], axis=1)
        # median = percentiles[0]
        # lower = median - percentiles[1]
        # upper = percentiles[2] - median

        # results_str = "".join([f" {median[i]:>15.10f} {lower[i]:>15.10f} {upper[i]:>15.10f}" for i in range(median.size)])
        # output = f"{output}{results_str}\n"

        output_dict["params"] = params

        cls.stats = {}
        cls.stats["params"] = params
        cls.stats["chain"] = chain
        cls.stats["log_prob"] = log_prob
        cls.stats["reduced_chain"] = reduced_chain
        cls.stats["reduced_log_prob"] = reduced_log_prob
        cls.stats_flag = True

        # TODO: Output to file. Can be based on dict
        # --------------------------------------------------------------------------
        # output = f"{lc_file.stem:>13s}"
        # output = f"{output} {str(geweke_flag):>7s} {num_walkers:>4d} {starts[start_index]/chain.shape[1]:>5.2f}"
        # output = f"{output} {r_mvar.max():>9.5f}{r_hat_str}"
        # results_str = "".join([f"{params['mapv'][i]:>15.10f} {params['modes'][i]:>15.10f} {params['median'][i]:>15.10f} {params['hpd_down'][i]:>15.10f} {params['hpd_up'][i]:>15.10f} {params['hpd_99_interval'][i]:>15.10f}" for i in range(params['median'].size)])
        # output = f"{output}{results_str}\n"

        # # Output to file
        # if fout is not None:
        #     log.info(f"Writing output to file {fout}")
        #     with open(fout, "a+") as o:
        #         o.write(output)
        # --------------------------------------------------------------------------

    @classmethod
    def analysis(cls, fout=None):
        if not cls.stats_flag:
            cls.statistics()
        
        if not cls.figdir.is_dir():
            cls.figdir.mkdir()      

        cls.plot_gelman_rubin()
        cls.plot_parameter_hist()
        cls.plot_corner_plot()
        cls.plot_log_prob_hist()
        cls.plot_trace_plot()
        cls.plot_lc_double_plot()
        cls.plot_psd_plot()
        plt.close("all")

    @classmethod
    def plot_gelman_rubin(cls, save=True):
        # TODO: This flag should not be in all functions but it will do for now
        if not cls.stats_flag:
            cls.statistics()
        log.info(f"Plotting Gelman-Rubin analysis")
        gelman_fig = gelman_rubin_plot(cls.stats["reduced_chain"], pnames=cls.stats["params"]["names"])
        if save:
            gelman_fig.savefig(f"{cls.figdir}/gelman_plot.pdf")
            plt.close(gelman_fig)
        else:
            return gelman_fig


    @classmethod
    def plot_parameter_hist(cls, save=True):
        if not cls.stats_flag:
            cls.statistics()
        log.info(f"Plotting parameter histograms")
        parameter_fig = parameter_hist(cls.stats["chain"], cls.stats["params"], pnames=cls.stats["params"]["names"])
        if save:
            parameter_fig.savefig(f"{cls.figdir}/parameter_hist.pdf")
            plt.close(parameter_fig)
        else:
            return parameter_fig

    @classmethod
    def plot_corner_plot(cls, save=True):
        if not cls.stats_flag:
            cls.statistics()
        log.info(f"Plotting corner")
        corner_fig = corner_plot(cls.stats["reduced_chain"], pnames=cls.stats["params"]["names"], downsample=5)
        if save:
            corner_fig.savefig(f"{cls.figdir}/corner_plot.pdf")
            plt.close(corner_fig)
        else:
            return corner_fig
    
    @classmethod
    def plot_log_prob_hist(cls, save=True):
        if not cls.stats_flag:
            cls.statistics()
        log.info(f"Plotting log_prob histogram")
        log_prob_fig = log_prob_hist(cls.stats["reduced_log_prob"])
        if save:
            log_prob_fig.savefig(f"{cls.figdir}/log_prob_hist.pdf")
            plt.close(log_prob_fig)
        else:
            return log_prob_fig

    @classmethod
    def plot_trace_plot(cls, save=True):
        if not cls.stats_flag:
            cls.statistics()
        log.info(f"Plotting traces")
        trace_fig = trace_plot(cls.stats["chain"], cls.stats["log_prob"], pnames=cls.stats["params"]["names"], downsample=10)
        if save:
            trace_fig.savefig(f"{cls.figdir}/trace_plot.pdf")
            plt.close(trace_fig)
        else:
            return trace_fig

    @classmethod
    def plot_lc_double_plot(cls, save=True):
        if not cls.stats_flag:
            cls.statistics()
        log.info(f"Plotting lightcurve with zoom")
        lc_double_fig = lc_double_plot(cls.gp_model, cls.mean_model, cls.stats["params"], cls.lc.time, cls.lc.flux, cls.lc.flux_err, offset=0.14, oversample=2)
        if save:
            lc_double_fig.savefig(f"{cls.figdir}/lc_double_plot.pdf")
            plt.close(lc_double_fig)
        else:
            return lc_double_fig
    
    @classmethod
    def plot_psd_plot(cls, save=True):
        if not cls.stats_flag:
            cls.statistics()
        log.info(f"Plotting PSD")
        psd_fig = psd_plot(cls.gp_model, cls.mean_model, cls.stats["params"], cls.lc.time, cls.lc.flux)
        if save:
            psd_fig.savefig(f"{cls.figdir}/psd_plot.pdf")
            plt.close(psd_fig)
        else:
            return psd_fig