"""Sampling-hyperparameter searches over eta, omega and the time distortion.

Entered from ``GraphDiscreteFlowModel.on_test_epoch_end`` when
``cfg.sample.search`` is set. Each ``search_*`` method is a search loop; the
plumbing they run on lives in :mod:`search.search_utils`. 
"""

import itertools
import os
import random
import time

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from search.search_utils import SearchUtilsMixin, objective_spec


class HyperparameterSearchMixin(SearchUtilsMixin):
    """The ``search_*`` entry points, dispatched by ``search_hyperparameters``."""

    def search_hyperparameters(self):
        """Run the search named by ``cfg.sample.search``."""
        num_step_list = [50]

        if self.cfg.dataset.name == "qm9":
            num_step_list = [1, 5, 10, 50, 100, 500]
        if self.cfg.dataset.name in ["guacamol", 'moses']:  # accelerate
            num_step_list = [50]

        searches = {
            "distortion": self.search_distortion,
            "stochasticity": self.search_stochasticity,
            "target_guidance": self.search_target_guidance,
            "full_grid": self.search_full_grid,
            "random": self.search_random,
            "sobol": self.search_sobol,
            "bo": self.search_bayesian_optimization,
            "fixed_configs": self.search_fixed_configs,
        }
        if self.cfg.sample.search == "all":
            to_run = ["distortion", "stochasticity", "target_guidance"]
        elif self.cfg.sample.search in searches:
            to_run = [self.cfg.sample.search]
        else:
            raise NotImplementedError(
                f"Search type {self.cfg.sample.search} not implemented."
            )

        self._search_started_at = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        self._search_summary_info = []
        self._write_search_summary()

        search_times = {}
        for name in to_run:
            t0 = time.time()
            searches[name](num_step_list)
            search_times[name] = time.time() - t0
        search_times["total"] = sum(search_times.values())
        self._write_search_summary(search_times)

        print("Finished searching. Timings in search_summary.txt")

    def search_distortion(self, num_step_list):
        """Grid search over the time distortions."""
        results_df = pd.DataFrame()
        distortion_list = ["identity", "polydec", "cos", "revcos", "polyinc"]
        seed_list = [0, 1, 2]

        out_dir = self._axis_search_dir("distortion")
        results_path = os.path.join(out_dir, "search_distortion.csv")

        for seed in seed_list:
            pl.seed_everything(seed)

            for num_step in num_step_list:
                for distortor in distortion_list:
                    self.cfg.sample.sample_steps = num_step
                    self.cfg.sample.time_distortion = distortor
                    print(
                        f"############# Testing num steps: {num_step}, distortor: {distortor}, seed: {seed} #############"
                    )
                    samples, labels, res, config_time = self._sample_and_evaluate()
                    res_df = self._result_row(res, num_step=num_step, distortor=distortor, seed=seed, time_s=config_time)
                    results_df = pd.concat([results_df, res_df], ignore_index=True)
                    results_df.to_csv(results_path)

        self.cfg.sample.time_distortion = "identity"

        results_df.reset_index(drop=True, inplace=True)
        results_df.set_index(["num_step", "distortor", "seed"], inplace=True)
        results_df.to_csv(results_path)

    def search_stochasticity(self, num_step_list):
        """Grid search over the stochasticity level eta."""
        results_df = pd.DataFrame()
        eta_list = [0.0, 5, 10, 25, 50, 100, 200, 300, 500]
        seed_list = [0, 1, 2]

        out_dir = self._axis_search_dir("stochasticity")
        results_path = os.path.join(out_dir, "search_stochasticity.csv")

        for seed in seed_list:
            pl.seed_everything(seed)

            for num_step in num_step_list:
                for eta in eta_list:
                    self.cfg.sample.sample_steps = num_step
                    self.cfg.sample.eta = eta
                    self.rate_matrix_designer.eta = eta
                    print(
                        f"############# Testing num steps: {num_step}, eta: {eta}, seed: {seed} #############"
                    )
                    samples, labels, res, config_time = self._sample_and_evaluate()
                    res_df = self._result_row(res, num_step=num_step, eta=eta, seed=seed, time_s=config_time)
                    results_df = pd.concat([results_df, res_df], ignore_index=True)
                    results_df.to_csv(results_path)

        self.cfg.sample.eta = 0.0
        self.rate_matrix_designer.eta = 0.0

        results_df.reset_index(drop=True, inplace=True)
        results_df.set_index(["num_step", "eta", "seed"], inplace=True)
        results_df.to_csv(results_path)

    def search_target_guidance(self, num_step_list):
        """Grid search over the target guidance omega."""
        results_df = pd.DataFrame()
        omega_list = [
            0.0,
            0.01,
            0.02,
            0.05,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            1.0,
            2.0,
        ]
        seed_list = [0, 1, 2]

        out_dir = self._axis_search_dir("target_guidance")
        results_path = os.path.join(out_dir, "search_target_guidance.csv")

        for seed in seed_list:
            pl.seed_everything(seed)

            for num_step in num_step_list:
                for omega in omega_list:
                    self.cfg.sample.sample_steps = num_step
                    self.cfg.sample.omega = omega
                    self.rate_matrix_designer.omega = omega
                    print(
                        f"############# Testing num steps: {num_step}, omega: {omega}, seed: {seed} #############"
                    )
                    samples, labels, res, config_time = self._sample_and_evaluate()
                    res_df = self._result_row(res, num_step=num_step, omega=omega, seed=seed, time_s=config_time)
                    results_df = pd.concat([results_df, res_df], ignore_index=True)
                    results_df.to_csv(results_path)

        self.cfg.sample.omega = 0.0
        self.rate_matrix_designer.omega = 0.0

        results_df.reset_index(drop=True, inplace=True)
        results_df.set_index(["num_step", "omega", "seed"], inplace=True)
        results_df.to_csv(results_path)

    def search_full_grid(self, num_step_list):
        """Grid search over distortion x eta x omega, checkpointed after every config."""
        distortion_list = ["identity", "polydec", "cos", "revcos", "polyinc"]
        eta_list = [0.0, 5, 10, 25, 50, 100]
        omega_list = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

        version_dir = self._search_version_dir("full_grid")
        checkpoint_path = os.path.join(version_dir, "results.csv")
        key_cols = ["num_step", "distortor", "eta", "omega"]
        dtypes = {"num_step": int, "distortor": str, "eta": float, "omega": float}
        results_df, completed = self._load_search_checkpoint(checkpoint_path, key_cols, dtypes)

        grid = itertools.product(num_step_list, distortion_list, eta_list, omega_list)
        todo = [c for c in grid if (int(c[0]), str(c[1]), float(c[2]), float(c[3])) not in completed]

        search_start = time.time()
        for run_idx, (num_step, distortor, eta, omega) in enumerate(todo, 1):
            self._apply_sampling_config(num_step, distortor, eta, omega)
            print(
                f"############# [{run_idx}/{len(todo)}] Testing num steps: {num_step}, "
                f"distortor: {distortor}, eta: {eta}, omega: {omega} #############"
            )
            samples, labels, res, config_time = self._sample_and_evaluate()
            self._print_progress(config_time, run_idx, len(todo), search_start)
            res_df = self._result_row(res, num_step=num_step, distortor=distortor, eta=eta, omega=omega, time_s=config_time)
            results_df = pd.concat([results_df, res_df], ignore_index=True)
            self._save_search_checkpoint(results_df, checkpoint_path)

        self._reset_sampling_config()

        results_df.reset_index(drop=True, inplace=True)
        results_df.set_index(key_cols, inplace=True)
        self._save_search_checkpoint(results_df, checkpoint_path)
        self._mark_search_done(version_dir)
        print(f"search_full_grid results checkpointed at {checkpoint_path}")

    def search_random(self, num_step_list):
        """Random search over distortion x eta x omega, checkpointed after every trial."""
        distortion_list = ["identity", "polydec", "cos", "revcos", "polyinc"]
        eta_low, eta_high = self.cfg.sample.search_eta_range
        omega_low, omega_high = self.cfg.sample.search_omega_range
        n_trials = self.cfg.sample.search_n_trials

        version_dir = self._search_version_dir(
            "random", tags=(f"seed{self.cfg.sample.search_seed}",)
        )
        checkpoint_path = os.path.join(version_dir, "results.csv")
        results_df, completed = self._load_search_checkpoint(
            checkpoint_path, ["num_step", "trial_idx"], {"num_step": int, "trial_idx": int}
        )

        rng = random.Random(self.cfg.sample.search_seed)
        n_total = len(num_step_list) * n_trials
        n_todo = n_total - len(completed)

        search_start = time.time()
        executed = 0
        for num_step, trial_idx in itertools.product(num_step_list, range(n_trials)):
            # drawn for every trial, completed or not, to keep the RNG stream aligned
            distortor = rng.choice(distortion_list)
            eta = rng.uniform(eta_low, eta_high)
            omega_raw = rng.uniform(omega_low, omega_high)
            omega = self._omega_power_transform(omega_raw)
            if (int(num_step), int(trial_idx)) in completed:
                continue

            executed += 1
            self._apply_sampling_config(num_step, distortor, eta, omega)
            print(
                f"############# [{executed}/{n_todo}] Random trial: num_steps: {num_step}, "
                f"distortor: {distortor}, eta: {eta:.4f}, omega: {omega:.4f} #############"
            )
            samples, labels, res, config_time = self._sample_and_evaluate()
            self._print_progress(config_time, executed, n_todo, search_start)
            # omega is what was sampled with, omega_raw the draw before the power transform
            res_df = self._result_row(res, num_step=num_step, distortor=distortor, eta=eta, omega=omega, omega_raw=omega_raw, trial_idx=trial_idx, time_s=config_time)
            results_df = pd.concat([results_df, res_df], ignore_index=True)
            self._save_search_checkpoint(results_df, checkpoint_path)

        self._reset_sampling_config()

        results_df.reset_index(drop=True, inplace=True)
        results_df.set_index(["num_step", "distortor", "eta", "omega"], inplace=True)
        self._save_search_checkpoint(results_df, checkpoint_path)
        self._mark_search_done(version_dir)
        print(f"search_random results checkpointed at {checkpoint_path}")

    def search_sobol(self, num_step_list):
        """Sobol (scrambled QMC) search over distortion x eta x omega."""
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        distortion_list = ["identity", "polydec", "cos", "revcos", "polyinc"]
        eta_low, eta_high = self.cfg.sample.search_eta_range
        omega_low, omega_high = self.cfg.sample.search_omega_range
        n_trials = self.cfg.sample.search_n_trials
        objective_cols, directions = objective_spec(
            self.cfg.sample.search_objective, self.cfg.dataset.directed
        )
        search_space = {
            "eta": optuna.distributions.FloatDistribution(eta_low, eta_high),
            "omega": optuna.distributions.FloatDistribution(omega_low, omega_high),
            "distortion_u": optuna.distributions.FloatDistribution(0, len(distortion_list)),
        }

        version_dir = self._search_version_dir(
            "sobol", tags=(f"seed{self.cfg.sample.search_seed}",)
        )
        checkpoint_path = os.path.join(version_dir, "results.csv")
        results_df, completed = self._load_search_checkpoint(
            checkpoint_path, ["num_step", "trial_idx"], {"num_step": int, "trial_idx": int}
        )

        n_todo = len(num_step_list) * n_trials - len(completed)
        search_start = time.time()
        executed = 0
        for num_step in num_step_list:
            study = optuna.create_study(
                sampler=optuna.samplers.QMCSampler(
                    qmc_type="sobol", scramble=True, seed=self.cfg.sample.search_seed
                ),
                **({"direction": directions[0]} if len(directions) == 1 else {"directions": directions}),
            )
            # the first ask has no completed trial to infer the space from and falls
            # back to independent sampling, so burn it
            study.tell(self._ask(study, search_space), state=optuna.trial.TrialState.PRUNED)
            n_done = self._replay_trials(
                study, results_df, num_step, search_space,
                {"eta": "eta", "omega": "omega_raw", "distortion_u": "distortion_u"},
                objective_cols, "sobol",
            )

            for trial_idx in range(n_done, n_trials):
                trial = self._ask(study, search_space)
                eta = float(trial.params["eta"])
                omega_raw = float(trial.params["omega"])
                omega = self._omega_power_transform(omega_raw)
                distortor = distortion_list[
                    min(int(trial.params["distortion_u"]), len(distortion_list) - 1)
                ]

                executed += 1
                self._apply_sampling_config(num_step, distortor, eta, omega)
                print(
                    f"############# [{executed}/{n_todo}] Sobol trial: num_steps: {num_step}, "
                    f"distortor: {distortor}, eta: {eta:.4f}, omega: {omega:.4f} #############"
                )
                samples, labels, res, config_time = self._sample_and_evaluate()
                self._print_progress(config_time, executed, n_todo, search_start)
                # omega is what was sampled with, omega_raw the draw before the power transform
                res_df = self._result_row(res, num_step=num_step, distortor=distortor, eta=eta, omega=omega, omega_raw=omega_raw, distortion_u=trial.params["distortion_u"], trial_idx=trial_idx, time_s=config_time)
                missing = [c for c in objective_cols if c not in res_df]
                if missing:
                    raise KeyError(
                        f"sample.search_objective '{self.cfg.sample.search_objective}' needs "
                        f"column(s) {missing}, not among the evaluated metrics: {sorted(res_df.columns)}"
                    )

                values = [float(res_df[col].iloc[0]) for col in objective_cols]
                if all(np.isfinite(v) for v in values):
                    study.tell(trial, values[0] if len(values) == 1 else values)
                else:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                results_df = pd.concat([results_df, res_df], ignore_index=True)
                self._save_search_checkpoint(results_df, checkpoint_path)

            if self.cfg.sample.search_visualize:
                self._save_optuna_visualizations(study, num_step, "sobol")

        self._reset_sampling_config()

        results_df.reset_index(drop=True, inplace=True)
        results_df.set_index(["num_step", "distortor", "eta", "omega"], inplace=True)
        self._save_search_checkpoint(results_df, checkpoint_path)
        self._mark_search_done(version_dir)
        print(f"search_sobol results checkpointed at {checkpoint_path}")

    def search_bayesian_optimization(self, num_step_list):
        """Bayesian optimization over distortion x eta x omega."""
        import optuna

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        sampler_name = self.cfg.sample.search_bo_sampler
        mode = self.cfg.sample.search_bo_distortion_mode
        n_trials = self.cfg.sample.search_n_trials
        objective_cols, directions = objective_spec(
            self.cfg.sample.search_objective, self.cfg.dataset.directed
        )
        eta_low, eta_high = self.cfg.sample.search_eta_range
        omega_low, omega_high = self.cfg.sample.search_omega_range
        search_space = {
            "eta": optuna.distributions.FloatDistribution(eta_low, eta_high),
            "omega": optuna.distributions.FloatDistribution(omega_low, omega_high),
            **self._bo_distortion_space(mode),
        }

        version_dir = self._search_version_dir(
            "bo",
            tags=(sampler_name, mode, f"seed{self.cfg.sample.search_seed}"),
        )
        checkpoint_path = os.path.join(version_dir, "results.csv")
        results_df, completed = self._load_search_checkpoint(
            checkpoint_path, ["num_step", "trial_idx"], {"num_step": int, "trial_idx": int}
        )
        param_cols = {"eta": "eta", "omega": "omega"}
        param_cols.update(
            {"distortion_a": "distortion_a", "distortion_b": "distortion_b"}
            if mode == "continuous"
            else {"time_distortion": "distortor"}
        )

        n_todo = len(num_step_list) * n_trials - len(completed)
        search_start = time.time()
        executed = 0
        best = {}
        for num_step in num_step_list:
            study = optuna.create_study(
                sampler=self._make_bo_sampler(
                    sampler_name, self.cfg.sample.search_bo_n_startup_trials
                ),
                **({"direction": directions[0]} if len(directions) == 1 else {"directions": directions}),
            )
            n_done = self._replay_trials(
                study, results_df, num_step, search_space, param_cols, objective_cols, sampler_name
            )

            for trial_idx in range(n_done, n_trials):
                trial = self._ask(study, search_space)
                eta = float(trial.params["eta"])
                omega = float(trial.params["omega"])
                distortion_label, distortion_values = self._bo_apply_distortion(trial.params, mode)
                self._apply_sampling_config(num_step=num_step, eta=eta, omega=omega)

                executed += 1
                print(
                    f"############# [{executed}/{n_todo}] BO trial ({sampler_name}): "
                    f"num_steps: {num_step}, {distortion_label}, eta: {eta:.4f}, "
                    f"omega: {omega:.4f} #############"
                )
                samples, labels, res, config_time = self._sample_and_evaluate()
                self._print_progress(config_time, executed, n_todo, search_start)
                res_df = self._result_row(res, num_step=num_step, **distortion_values, eta=eta, omega=omega, trial_idx=trial_idx, time_s=config_time)
                missing = [c for c in objective_cols if c not in res_df]
                if missing:
                    raise KeyError(
                        f"sample.search_objective '{self.cfg.sample.search_objective}' needs "
                        f"column(s) {missing}, not among the evaluated metrics: {sorted(res_df.columns)}"
                    )

                values = [float(res_df[col].iloc[0]) for col in objective_cols]
                if all(np.isfinite(v) for v in values):
                    study.tell(trial, values[0] if len(values) == 1 else values)
                else:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)
                results_df = pd.concat([results_df, res_df], ignore_index=True)
                self._save_search_checkpoint(results_df, checkpoint_path)

            # one best trial for a single objective, the whole Pareto front for 'both'
            best[num_step] = [(t.values, t.params) for t in study.best_trials]
            if self.cfg.sample.search_visualize:
                self._save_optuna_visualizations(study, num_step, sampler_name)

        self._reset_sampling_config()

        self._search_summary_info += [
            f"sampler: {sampler_name} (distortion mode: {mode})",
            f"objective: {self.cfg.sample.search_objective} -> "
            + ", ".join(f"{c} ({d})" for c, d in zip(objective_cols, directions)),
        ] + [
            f"best[num_step={ns}]: "
            + ", ".join(f"{c}={v:.6f}" for c, v in zip(objective_cols, values))
            + " at " + ", ".join(f"{k}={v}" for k, v in params.items())
            for ns, trials in best.items()
            for values, params in trials
        ]

        distortion_cols = ("distortion_a", "distortion_b") if mode == "continuous" else ("distortor",)
        results_df.reset_index(drop=True, inplace=True)
        results_df.set_index(["num_step", *distortion_cols, "eta", "omega"], inplace=True)
        self._save_search_checkpoint(results_df, checkpoint_path)
        self._mark_search_done(version_dir)
        print(f"search_bayesian_optimization results checkpointed at {checkpoint_path}")

    def search_fixed_configs(self, _num_step_list=None):
        """Evaluate a fixed list of sampling configs read from a CSV.

        The step counts and seeds are fixed here rather than taken from
        ``search_hyperparameters``, so a config list is always evaluated on the
        same grid. Checkpointed after every run.
        """
        num_step_list = [5, 10, 25, 50, 100, 250, 500, 1000]
        seed_list = [0, 1, 2]
        configs, csv_path, csv_header = self._load_fixed_configs()

        version_dir = self._search_version_dir("fixed_configs")
        self._check_fixed_configs_snapshot(version_dir, csv_path)
        checkpoint_path = os.path.join(version_dir, "results.csv")
        stats_path = os.path.join(version_dir, "seed_stats.csv")
        key_cols = ["config_idx", "num_step", "seed"]
        dtypes = {"config_idx": int, "num_step": int, "seed": int}
        results_df, completed = self._load_search_checkpoint(checkpoint_path, key_cols, dtypes)

        # one row per config, holding every column of the source CSV, to be joined
        # back onto the per-config aggregates
        configs_df = pd.DataFrame(
            [dict(config["row"], config_idx=idx) for idx, config in enumerate(configs)]
        )
        # the fields that describe a config rather than define it (method, objective)
        descriptive = [c for c in csv_header if c not in ("time", *self.FIXED_CONFIG_NUMERIC)]

        grid = itertools.product(range(len(configs)), num_step_list, seed_list)
        todo = [c for c in grid if (int(c[0]), int(c[1]), int(c[2])) not in completed]
        print(
            f"Evaluating {len(configs)} fixed config(s) at num_steps {list(num_step_list)}, "
            f"seeds {seed_list} from {csv_path}"
        )

        search_start = time.time()
        for run_idx, (config_idx, num_step, seed) in enumerate(todo, 1):
            config = configs[config_idx]
            pl.seed_everything(seed)
            self._apply_sampling_config(
                num_step,
                config["distortor"],
                config["eta"],
                config["omega"],
                config["a"],
                config["b"],
            )
            label = "/".join(str(config["row"][c]) for c in descriptive)
            print(
                f"############# [{run_idx}/{len(todo)}] Fixed config {config_idx} ({label}): "
                f"num_steps: {num_step}, distortor: {config['distortor']}, "
                f"eta: {config['eta']:.4f}, omega: {config['omega']:.4f}, "
                f"a: {config['a']:.4f}, b: {config['b']:.4f}, seed: {seed} #############"
            )
            samples, labels, res, config_time = self._sample_and_evaluate()
            self._print_progress(config_time, run_idx, len(todo), search_start)
            res_df = self._result_row(
                res,
                config_idx=config_idx,
                num_step=num_step,
                seed=seed,
                distortor=config["distortor"],
                eta=config["eta"],
                omega=config["omega"],
                distortion_a=config["a"],
                distortion_b=config["b"],
                time_s=config_time,
                **{c: config["row"][c] for c in descriptive},
            )
            results_df = pd.concat([results_df, res_df], ignore_index=True)
            self._save_search_checkpoint(results_df, checkpoint_path)
            self._save_fixed_configs_seed_stats(results_df, configs_df, csv_header, stats_path)

        self._reset_sampling_config()

        self._search_summary_info += [
            f"configs_csv: {csv_path}",
            f"n_configs: {len(configs)}",
            f"num_steps: {list(num_step_list)}",
            f"seeds: {seed_list}",
        ]

        self._save_fixed_configs_seed_stats(results_df, configs_df, csv_header, stats_path)
        results_df.reset_index(drop=True, inplace=True)
        results_df.set_index(key_cols, inplace=True)
        self._save_search_checkpoint(results_df, checkpoint_path)
        self._mark_search_done(version_dir)
        print(
            f"search_fixed_configs results checkpointed at {checkpoint_path} "
            f"(per-seed aggregates at {stats_path})"
        )
