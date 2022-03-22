"""
.. module:: samplers.polychord

:Synopsis: Interface for the PolyChord nested sampler
:Author: Will Handley, Mike Hobson and Anthony Lasenby (for PolyChord),
         Jesus Torrado (for the cobaya wrapper only)
"""
# Global
import os
import sys
import numpy as np
import logging
import inspect
from itertools import chain
from typing import Any, Callable, Optional
from tempfile import gettempdir
import re
import os
# Local
from cobaya.tools import read_dnumber, get_external_function, \
    find_with_regexp, NumberWithUnits, load_module, VersionCheckError
from cobaya.sampler import Sampler
from cobaya.mpi import is_main_process, share_mpi, sync_processes
from cobaya.collection import SampleCollection
from cobaya.log import LoggedError, get_logger
from cobaya.install import download_github_release, NotInstalledError
from cobaya.yamll import yaml_dump_file
from cobaya.conventions import derived_par_name_separator, packages_path_arg, Extension
from scipy import stats


class polychord(Sampler):
    r"""
    PolyChord sampler \cite{Handley:2015fda,2015MNRAS.453.4384H}, a nested sampler
    tailored for high-dimensional parameter spaces with a speed hierarchy.
    """
    # Name of the PolyChord repo and version to download
    _pc_repo_name = "PolyChord/PolyChordLite"
    _pc_repo_version = "1.18.2"
    _base_dir_suffix = "polychord_raw"
    _clusters_dir = "clusters"
    _at_resume_prefer_old = Sampler._at_resume_prefer_old + ["blocking"]
    _at_resume_prefer_new = Sampler._at_resume_prefer_new + ["callback_function"]
    pypolychord: Any

    # variables from yaml
    do_clustering: bool
    num_repeats: int
    confidence_for_unbounded: float
    callback_function: Callable
    blocking: Any
    measure_speeds: bool
    oversample_power: float
    nlive: NumberWithUnits
    path: str

    def get_proposal(self, sampler):
        if sampler == 0:  # 0th sampler is by default Polychord
            f = open("chains/test.stats", "r")
            lines = f.readlines()
            mu = []
            sig = []
            for entry in lines[30: 30 + self.n_sampled]:
                mu.append(float(entry.split()[1]))
                sig.append(float(entry.split()[3]))
        elif sampler == 1:
            from anesthetic import MCMCSamples
            root = "icelake/camb_default"
            planck_samples = MCMCSamples(root=root)
            mu = planck_samples.mean().values[:27]
            sig = np.diag(planck_samples.cov().values)[:27]
        return mu, sig

    def initialize(self):
        """Imports the PolyChord sampler and prepares its arguments."""
        # Allow global import if no direct path specification
        allow_global = not self.path
        if not self.path and self.packages_path:
            self.path = self.get_path(self.packages_path)
        self.pc: Any = self.is_installed(path=self.path, allow_global=allow_global,
                                         check=False)
        if not self.pc:
            raise NotInstalledError(
                self.log, "Could not find PolyChord. Check error message above. "
                          "To install it, run 'cobaya-install polychord --%s "
                          "[packages_path]'", packages_path_arg)
        # Prepare arguments and settings
        self.reorder = False
        self.n_hyperparam = {"normal": 0, "beta": 1, "gamma": 2, "delta": 2}
        self.n_sampled = len(self.model.parameterization.sampled_params())
        self.n_derived = len(self.model.parameterization.derived_params())
        self.n_priors = len(self.model.prior)
        self.n_likes = len(self.model.likelihood)
        if self.proposal_mode is not None:
            self.nDims = self.model.prior.d() + self.n_hyperparam[self.proposal_mode]
        else:
            self.nDims = self.model.prior.d()
        self.nDerived = (self.n_derived + self.n_priors + self.n_likes)
        if self.logzero is None:
            self.logzero = np.nan_to_num(-np.inf)
        if self.max_ndead == np.inf:
            self.max_ndead = -1
        self._quants_d_units = ["nlive", "max_ndead"]
        for p in self._quants_d_units:
            if getattr(self, p) is not None:
                setattr(self, p, NumberWithUnits(
                    getattr(self, p), "d", scale=self.nDims, dtype=int).value)
        self._quants_nlive_units = ["nprior"]
        for p in self._quants_nlive_units:
            if getattr(self, p) is not None:
                setattr(self, p, NumberWithUnits(
                    getattr(self, p), "nlive", scale=self.nlive, dtype=int).value)
        # Fill the automatic ones
        if getattr(self, "feedback", None) is None:
            values = {logging.CRITICAL: 0, logging.ERROR: 0, logging.WARNING: 0,
                      logging.INFO: 1, logging.DEBUG: 2}
            self.feedback = values[self.log.getEffectiveLevel()]
        # Prepare output folders and prefixes
        if self.output:
            self.file_root = self.output.prefix
            self.read_resume = self.output.is_resuming()
        else:
            output_prefix = share_mpi(hex(int(self._rng.random() * 16 ** 6))[2:]
                                      if is_main_process() else None)
            self.file_root = output_prefix
            # dummy output -- no resume!
            self.read_resume = False
        self.base_dir = self.get_base_dir(self.output)
        self.raw_clusters_dir = os.path.join(self.base_dir, self._clusters_dir)
        self.output.create_folder(self.base_dir)
        if self.do_clustering:
            self.clusters_folder = self.get_clusters_dir(self.output)
            self.output.create_folder(self.clusters_folder)
        self.mpi_info("Storing raw PolyChord output in '%s'.", self.base_dir)
        # Exploiting the speed hierarchy
        if self.blocking:
            blocks, oversampling_factors = self.model.check_blocking(self.blocking)
        else:
            if self.measure_speeds:
                self.model.measure_and_set_speeds(n=self.measure_speeds,
                                                  random_state=self._rng)
            blocks, oversampling_factors = self.model.get_param_blocking_for_sampler(
                oversample_power=self.oversample_power)
            if self.proposal_mode is not None:
                # blocks = np.append(blocks, [["beta"]], axis=0)
                blocks[0].insert(0, "beta")
                self.grade_dims = [len(block) for block in blocks]
                # self.grade_dims.append(1)
                # oversampling_factors = np.append(oversampling_factors, 1)
            else:
                self.grade_dims = [len(block) for block in blocks]
        self.mpi_info("Parameter blocks and their oversampling factors:")
        max_width = len(str(max(oversampling_factors)))
        for f, b in zip(oversampling_factors, blocks):
            self.mpi_info("* %" + "%d" % max_width + "d : %r", f, b)
        # Save blocking in updated info, in case we want to resume
        self._updated_info["blocking"] = list(zip(oversampling_factors, blocks))
        blocks_flat = list(chain(*blocks))
        self.ordering = [
            blocks_flat.index(p) for p in self.model.parameterization.sampled_params()]
        if self.proposal_mode is not None and self.n_hyperparam[self.proposal_mode] != 0:
            mu, sig = self.get_proposal(self.proposal_source)
            if self.reorder:
                self.mu = np.array([mu[ind] for ind in self.ordering])
                self.sig = np.array([sig[ind] for ind in self.ordering])
            else:
                self.mu = mu
                self.sig = sig
        # Steps per block
        # NB: num_repeats is ignored by PolyChord when int "grade_frac" given,
        # so needs to be applied by hand.
        # In num_repeats, `d` is interpreted as dimension of each block
        self.grade_frac = [
            int(o * read_dnumber(self.num_repeats, dim_block))
            for o, dim_block in zip(oversampling_factors, self.grade_dims)]
        # Assign settings
        pc_args = ["nlive", "num_repeats", "nprior", "do_clustering",
                   "precision_criterion", "max_ndead", "boost_posterior", "feedback",
                   "logzero", "posteriors", "equals", "compression_factor",
                   "cluster_posteriors", "write_resume", "read_resume", "write_stats",
                   "write_live", "write_dead", "base_dir",
                   "feedback", "read_resume", "base_dir", "file_root", "seed", "grade_dims", "grade_frac"]
        # TODO: Actually fix the dimensionality instead of deleting the pc_arg 'grade_dims'
        # As stated above, num_repeats is ignored, so let's not pass it
        pc_args.pop(pc_args.index("num_repeats"))
        settings: Any = load_module('pypolychord.settings', path=self._poly_build_path,
                                    min_version=None)
        self.pc_settings = settings.PolyChordSettings(
            self.nDims, self.nDerived, seed=(self.seed if self.seed is not None else -1),
            **{p: getattr(self, p) for p in pc_args if getattr(self, p) is not None})
        # Prepare callback function
        if self.callback_function is not None:
            self.callback_function_callable = (
                get_external_function(self.callback_function))
        self.last_point_callback = 0
        # Prepare runtime live and dead points collections
        self.live = SampleCollection(self.model, None, name="live")
        self.dead = SampleCollection(self.model, self.output, name="dead")
        # Done!
        if is_main_process():
            self.log.debug("Calling PolyChord with arguments:")
            for p, v in inspect.getmembers(self.pc_settings, lambda a: not (callable(a))):
                if not p.startswith("_"):
                    self.log.debug("  %s: %s", p, v)
        self.mpi_info("Initialized!")

    def dumper(self, live_points, dead_points, logweights, logZ, logZstd):
        if self.callback_function is None:
            return
        # Store live and dead points and evidence computed so far
        self.live.reset()
        for point in live_points:
            self.live.add(
                point[:self.n_sampled],
                derived=point[self.n_sampled:self.n_sampled + self.n_derived],
                weight=np.nan,
                logpriors=point[self.n_sampled + self.n_derived:
                                self.n_sampled + self.n_derived + self.n_priors],
                loglikes=point[self.n_sampled + self.n_derived + self.n_priors:
                               self.n_sampled + self.n_derived + self.n_priors +
                               self.n_likes])
        for logweight, point in zip(logweights[self.last_point_callback:],
                                    dead_points[self.last_point_callback:]):
            self.dead.add(
                point[:self.n_sampled],
                derived=point[self.n_sampled:self.n_sampled + self.n_derived],
                weight=np.exp(logweight),
                logpriors=point[self.n_sampled + self.n_derived:
                                self.n_sampled + self.n_derived + self.n_priors],
                loglikes=point[self.n_sampled + self.n_derived + self.n_priors:
                               self.n_sampled + self.n_derived + self.n_priors +
                               self.n_likes])
        self.logZ, self.logZstd = logZ, logZstd
        self._correct_unphysical_fraction()
        # Callback function
        if self.callback_function is not None:
            try:
                self.callback_function_callable(self)
            except Exception as e:
                self.log.error("The callback function produced an error: %r", str(e))
            self.last_point_callback = len(self.dead)

    def run(self):
        """
        Prepares the posterior function and calls ``PolyChord``'s ``run`` function.
        """

        if self.proposal_mode is not None:
            # bounds_unordered = self.model.prior.bounds()
            # if self.reorder:
            #     bounds = [bounds_unordered[ind] for ind in self.ordering]
            # else:
            #     bounds = bounds_unordered
            # lower = np.array([pair[0] for pair in bounds])
            # upper = np.array([pair[1] for pair in bounds])
            # for ind, bound in enumerate(lower):
            #     if np.isinf(bound):
            #         lower[ind] = self.mu[ind] - 5 * self.sig[ind]
            #         upper[ind] = self.mu[ind] + 5 * self.sig[ind]

            # Tightened prior bounds, will improve the convergence
            upper_new = self.mu + np.array([(np.sqrt(sig) if sig > 1 else sig) for sig in self.sig])
            lower_new = self.mu - np.array([(np.sqrt(sig) if sig > 1 else sig) for sig in self.sig])
            diff_new = upper_new - lower_new
            x_upper = np.array([self.model.prior.pdf[i].cdf(upper_new[i]) for i in range(self.n_sampled)])
            x_lower = np.array([self.model.prior.pdf[i].cdf(lower_new[i]) for i in range(self.n_sampled)])
            x_diff = x_upper - x_lower

        # Essentially what is needed here will be the sampled distribution means

        # Prepare the polychord likelihood
        def loglikelihood(params_values):
            if self.reorder:
                params_values = np.array([params_values[i] for i in np.argsort(self.ordering)])
            result = self.model.logposterior(params_values)
            loglikes = result.loglikes
            if len(loglikes) != self.n_likes:
                loglikes = np.full(self.n_likes, np.nan)
            derived = result.derived
            if len(derived) != self.n_derived:
                derived = np.full(self.n_derived, np.nan)
            derived = list(derived) + list(result.logpriors) + list(loglikes)
            return max(loglikes.sum(), self.pc_settings.logzero), derived

        def loglikelihood_tilde(theta_full):
            theta = theta_full[1:]
            beta = theta_full[0]
            if beta == 0:
                return loglikelihood(theta)
            else:
                return loglikelihood(theta)[0] + np.log(pi(theta) / pi_tilde(theta, beta)), loglikelihood(theta)[1]

        def pi_tilde(theta, beta):
            return np.product(
                beta * np.array([self.model.prior.pdf[i].pdf(theta[i]) for i in range(self.n_sampled)]) + (
                        1 - beta) * ((theta < upper_new) & (theta > lower_new)) / diff_new)

        def pi(theta):
            return np.product([self.model.prior.pdf[i].pdf(theta[i]) for i in range(self.n_sampled)])

        def prior(cube_full):
            if self.proposal_mode is not None:
                if self.reorder:
                    circle = np.array([cube_full[i] for i in self.ordering])
                else:
                    circle = cube_full[1:]
                beta = cube_full[0]

                if beta == 0:
                    cube = circle
                else:
                    cube = (circle < beta * x_lower) * (circle / beta) + (
                            (circle >= beta * x_lower) & (circle < beta * x_upper + (1 - beta))) * (
                                   circle + (1 - beta) * x_lower / x_diff) / (beta + (1 - beta) / x_diff) + (
                                   circle >= beta * x_upper + (1 - beta)) * (circle - (1 - beta)) / beta
                theta = np.array([self.model.prior.pdf[i].ppf(cube[i]) for i in range(self.n_sampled)])
                theta_full = np.empty_like(cube_full)
                theta_full[1:] = theta
                theta_full[0] = beta
                return theta_full
            else:
                cube = cube_full
                theta = np.empty_like(cube)
                for i, xi in enumerate(np.array(cube)[self.ordering]):
                    theta[i] = self.model.prior.pdf[i].ppf(xi)
                return theta

        if is_main_process():
            self.dump_paramnames(self.raw_prefix)
        sync_processes()
        self.mpi_info("Calling PolyChord...")
        if self.proposal_mode is not None:
            self.pc.run_polychord(loglikelihood_tilde, self.nDims, self.nDerived, self.pc_settings,
                                  prior, self.dumper)
        else:
            self.pc.run_polychord(loglikelihood, self.nDims, self.nDerived, self.pc_settings,
                                  prior, self.dumper)
        self.process_raw_output()

    @property
    def raw_prefix(self):
        return os.path.join(
            self.pc_settings.base_dir, self.pc_settings.file_root)

    def dump_paramnames(self, prefix):
        labels = self.model.parameterization.labels()
        with open(prefix + ".paramnames", "w") as f_paramnames:
            for p in self.model.parameterization.sampled_params():
                f_paramnames.write("%s\t%s\n" % (p, labels.get(p, "")))
            for p in self.model.parameterization.derived_params():
                f_paramnames.write("%s*\t%s\n" % (p, labels.get(p, "")))
            for p in self.model.prior:
                f_paramnames.write("%s*\t%s\n" % (
                    "logprior" + derived_par_name_separator + p,
                    r"\log\pi_\mathrm{" + p.replace("_", r"\ ") + r"}"))
            for p in self.model.likelihood:
                f_paramnames.write("%s*\t%s\n" % (
                    "loglike" + derived_par_name_separator + p,
                    r"\log\mathcal{L}_\mathrm{" + p.replace("_", r"\ ") + r"}"))

    def save_sample(self, fname, name):
        sample = np.atleast_2d(np.loadtxt(fname))
        if not sample.size:
            return None
        collection = SampleCollection(self.model, self.output, name=str(name))
        for row in sample:
            collection.add(
                row[2:2 + self.n_sampled],
                derived=row[2 + self.n_sampled:2 + self.n_sampled + self.n_derived],
                weight=row[0],
                logpriors=row[-(self.n_priors + self.n_likes):-self.n_likes],
                loglikes=row[-self.n_likes:])
        # make sure that the points are written
        collection.out_update()
        return collection

    def _correct_unphysical_fraction(self):
        """
        Correction for the fraction of the prior that is unphysical -- see issue #77
        """
        if not hasattr(self, "_frac_unphysical"):
            with open(self.raw_prefix + ".prior_info", "r", encoding="utf-8-sig") as pf:
                lines = list(pf.readlines())
            get_value_str = lambda line: line[line.find("=") + 1:]
            get_value_str_var = lambda var: get_value_str(
                next(line for line in lines if line.lstrip().startswith(var)))
            nprior = int(get_value_str_var("nprior"))
            ndiscarded = int(get_value_str_var("ndiscarded"))
            self._frac_unphysical = nprior / ndiscarded
        if self._frac_unphysical != 1:
            self.log.debug(
                "Correcting for unphysical region fraction: %g", self._frac_unphysical)
            self.logZ += np.log(self._frac_unphysical)
            if hasattr(self, "clusters"):
                for cluster in self.clusters.values():
                    cluster["logZ"] += np.log(self._frac_unphysical)

    def process_raw_output(self):
        """
        Loads the sample of live points from ``PolyChord``'s raw output and writes it
        (if ``txt`` output requested).
        """
        if is_main_process():
            self.log.info("Loading PolyChord's results: samples and evidences.")
            self.dump_paramnames(self.raw_prefix)
            self.collection = self.save_sample(self.raw_prefix + ".txt", "1")
            # Load clusters, and save if output
            if self.pc_settings.do_clustering:
                self.clusters = {}
                clusters_raw_regexp = re.compile(
                    re.escape(self.pc_settings.file_root + "_") + r"\d+\.txt")
                cluster_raw_files = sorted(find_with_regexp(
                    clusters_raw_regexp, os.path.join(
                        self.pc_settings.base_dir, self._clusters_dir), walk_tree=True))
                for f in cluster_raw_files:
                    i = int(f[f.rfind("_") + 1:-len(".txt")])
                    if self.output:
                        old_folder = self.output.folder
                        self.output.folder = self.clusters_folder
                    sample = self.save_sample(f, str(i))
                    if self.output:
                        # noinspection PyUnboundLocalVariable
                        self.output.folder = old_folder
                    self.clusters[i] = {"sample": sample}
            # Prepare the evidence(s) and write to file
            pre = "log(Z"
            active = "(Still active)"
            with open(self.raw_prefix + ".stats", "r", encoding="utf-8-sig") as statsfile:
                lines = [line for line in statsfile.readlines() if line.startswith(pre)]
            for line in lines:
                logZ, logZstd = [float(n.replace(active, "")) for n in
                                 line.split("=")[-1].split("+/-")]
                component = line.split("=")[0].lstrip(pre + "_").rstrip(") ")
                if not component:
                    self.logZ, self.logZstd = logZ, logZstd
                elif self.pc_settings.do_clustering:
                    i = int(component)
                    self.clusters[i]["logZ"], self.clusters[i]["logZstd"] = logZ, logZstd
            self.log.debug(
                "RAW log(Z) = %g +/- %g ; RAW Z in [%.8g, %.8g] (68%% C.L. log-gaussian)",
                self.logZ, self.logZstd,
                *[np.exp(self.logZ + n * self.logZstd) for n in [-1, 1]])
            self._correct_unphysical_fraction()
            if self.output:
                out_evidences = dict(logZ=self.logZ, logZstd=self.logZstd)
                if getattr(self, "clusters", None):
                    out_evidences["clusters"] = {}
                    for i in sorted(list(self.clusters)):
                        out_evidences["clusters"][i] = dict(
                            logZ=self.clusters[i]["logZ"],
                            logZstd=self.clusters[i]["logZstd"])
                fname = os.path.join(self.output.folder,
                                     self.output.prefix + Extension.evidence)
                yaml_dump_file(fname, out_evidences, comment="log-evidence",
                               error_if_exists=False)
        # TODO: try to broadcast the collections
        # if get_mpi():
        #     bcast_from_0 = lambda attrname: setattr(self,
        #         attrname, get_mpi_comm().bcast(getattr(self, attrname, None), root=0))
        #     map(bcast_from_0, ["collection", "logZ", "logZstd", "clusters"])
        if is_main_process():
            self.log.info("Finished! Raw PolyChord output stored in '%s', "
                          "with prefix '%s'",
                          self.pc_settings.base_dir, self.pc_settings.file_root)
            self.log.info(
                "log(Z) = %g +/- %g ; Z in [%.8g, %.8g] (68%% C.L. log-gaussian)",
                self.logZ, self.logZstd,
                *[np.exp(self.logZ + n * self.logZstd) for n in [-1, 1]])

    def products(self):
        """
        Auxiliary function to define what should be returned in a scripted call.

        Returns:
           The sample ``SampleCollection`` containing the sequentially
           discarded live points.
        """
        if is_main_process():
            products = {
                "sample": self.collection, "logZ": self.logZ, "logZstd": self.logZstd}
            if self.pc_settings.do_clustering:
                products.update({"clusters": self.clusters})
            return products
        else:
            return {}

    @classmethod
    def get_base_dir(cls, output):
        if output:
            return output.add_suffix(cls._base_dir_suffix, separator="_")
        return os.path.join(gettempdir(), cls._base_dir_suffix)

    @classmethod
    def get_clusters_dir(cls, output):
        if output:
            return output.add_suffix(cls._clusters_dir, separator="_")

    @classmethod
    def output_files_regexps(cls, output, info=None, minimal=False):
        # Resume file
        regexps_tuples = [
            (re.compile(re.escape(output.prefix + ".resume")), cls.get_base_dir(output))]
        if minimal:
            return regexps_tuples
        return regexps_tuples + [
            # Raw products base dir
            (None, cls.get_base_dir(output)),
            # Main sample
            (output.collection_regexp(name=None), None),
            # Evidence
            (re.compile(re.escape(output.prefix + Extension.evidence)), None),
            # Clusters
            (None, cls.get_clusters_dir(output))
        ]

    @classmethod
    def get_version(cls):
        return None

    @classmethod
    def get_path(cls, path):
        return os.path.realpath(
            os.path.join(path, "code",
                         cls._pc_repo_name[cls._pc_repo_name.find("/") + 1:]))

    @classmethod
    def get_import_path(cls, path):
        log = get_logger(cls.__name__)
        poly_build_path = os.path.join(path, "build")
        if not os.path.isdir(poly_build_path):
            log.error("Either PolyChord is not in the given folder, "
                      "'%s', or you have not compiled it.", path)
            return None
        py_version = "%d.%d" % (sys.version_info.major, sys.version_info.minor)
        try:
            post = next(d for d in os.listdir(poly_build_path)
                        if (d.startswith("lib.") and py_version in d))
        except StopIteration:
            log.error("The PolyChord installation at '%s' has not been compiled for the "
                      "current Python version.", path)
            return None
        return os.path.join(poly_build_path, post)

    @classmethod
    def is_compatible(cls):
        import platform
        if platform.system() == "Windows":
            return False
        return True

    @classmethod
    def is_installed(cls, **kwargs):
        log = get_logger(cls.__name__)
        if not kwargs.get("code", True):
            return True
        check = kwargs.get("check", True)
        func = log.info if check else log.error
        path: Optional[str] = kwargs["path"]
        if path is not None and path.lower() == "global":
            path = None
        if path and not kwargs.get("allow_global"):
            if is_main_process():
                log.info("Importing *local* PolyChord from '%s'.", path)
            if not os.path.exists(path):
                if is_main_process():
                    func("The given folder does not exist: '%s'", path)
                return False
            poly_build_path = cls.get_import_path(path)
            if not poly_build_path:
                return False
        elif not path:
            if is_main_process():
                log.info("Importing *global* PolyChord.")
            poly_build_path = None
        else:
            if is_main_process():
                log.info(
                    "Importing *auto-installed* PolyChord (but defaulting to *global*).")
            poly_build_path = cls.get_import_path(path)
        cls._poly_build_path = poly_build_path
        try:
            # TODO: add min_version when polychord module version available
            return load_module(
                'pypolychord', path=poly_build_path, min_version=None)
        except ModuleNotFoundError:
            if path is not None and path.lower() != "global":
                log.error("Couldn't find the PolyChord python interface at '%s'. "
                          "Are you sure it has been installed there?", path)
            elif not check:
                log.error("Could not import global PolyChord installation. "
                          "Specify a Cobaya or PolyChord installation path, "
                          "or install the PolyChord Python interface globally with "
                          "'cd /path/to/polychord/ ; python setup.py install'")
            return False
        except ImportError as e:
            log.error("Couldn't load the PolyChord python interface in %s:\n"
                      "%s", poly_build_path or "global", e)
            return False
        except VersionCheckError as e:
            log.error(str(e))
            return False

    @classmethod
    def install(cls, path=None, force=False, code=False, data=False,
                no_progress_bars=False):
        if not code:
            return True
        log = get_logger(__name__)
        log.info("Downloading PolyChord...")
        success = download_github_release(os.path.join(path, "code"), cls._pc_repo_name,
                                          cls._pc_repo_version,
                                          no_progress_bars=no_progress_bars,
                                          logger=log)
        if not success:
            log.error("Could not download PolyChord.")
            return False
        log.info("Compiling (Py)PolyChord...")
        from subprocess import Popen, PIPE
        # Needs to re-define os' PWD,
        # because MakeFile calls it and is not affected by the cwd of Popen
        cwd = os.path.join(path, "code",
                           cls._pc_repo_name[cls._pc_repo_name.find("/") + 1:])
        my_env = os.environ.copy()
        my_env.update({"PWD": cwd})
        process_make = Popen(["make", "pypolychord", "MPI=1"], cwd=cwd, env=my_env,
                             stdout=PIPE, stderr=PIPE)
        out, err = process_make.communicate()
        if process_make.returncode:
            log.info(out.decode("utf-8"))
            log.info(err.decode("utf-8"))
            log.error("Compilation failed!")
            return False
        my_env.update({"CC": "mpicc", "CXX": "mpicxx"})
        process_make = Popen([sys.executable, "setup.py", "build"],
                             cwd=cwd, env=my_env, stdout=PIPE, stderr=PIPE)
        out, err = process_make.communicate()
        if process_make.returncode:
            log.info(out.decode("utf-8"))
            log.info(err.decode("utf-8"))
            log.error("Python build failed!")
            return False
        return True
