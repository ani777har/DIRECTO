"""Support code for the searches in :mod:`search.hyperparameter_search`.

Output directories, resume/checkpoint I/O, the per-trial generate-and-evaluate
step and the Optuna study mechanics. Mixed into ``GraphDiscreteFlowModel``, so
``self`` is the model.
"""

import csv
import os
import time

import numpy as np
import pandas as pd
import torch
from hydra.utils import get_original_cwd


def objective_spec(objective, directed):
    """(result columns, optuna directions) for a sample.search_objective."""
    ratio = "ratio/average_ratio_mean"
    # the directed metrics prefix every key with the split, the undirected ones
    # hardcode 'sampling'; the searches always evaluate with test=True
    vun = ("test/" if directed else "sampling/") + "frac_unic_non_iso_valid_mean"
    specs = {
        "ratio": ([ratio], ["minimize"]),
        "vun": ([vun], ["maximize"]),
        "both": ([ratio, vun], ["minimize", "maximize"]),
    }
    if objective not in specs:
        raise ValueError(f"Unknown search_objective '{objective}'. Choose from {list(specs)}.")
    return specs[objective]


class SearchUtilsMixin:
    """Helpers shared by the sampling-hyperparameter searches."""

    def _sample_and_evaluate(self):
        """Generate and evaluate one sampling configuration. Returns (samples, labels, res, total_time)"""
        t0 = time.time()
        samples, labels = self.sample(
            is_test=True,
            save_samples=self.cfg.general.save_samples,
            save_visualization=False,
        )
        sample_time = time.time() - t0

        t1 = time.time()
        res = self.evaluate_samples(samples=samples, labels=labels, is_test=True)
        eval_time = time.time() - t1

        res["sampling_time_s"] = (sample_time, 0.0)
        res["eval_time_s"] = (eval_time, 0.0)
        print(
            f"  -> generation {sample_time:.2f}s | evaluation {eval_time:.2f}s "
            f"(eval/gen = {eval_time / max(sample_time, 1e-9):.2f})"
        )
        return samples, labels, res, sample_time + eval_time

    @staticmethod
    def _result_row(res, **cols):
        row = {f"{key}_mean": value[0] for key, value in res.items()}
        row.update({f"{key}_std": value[1] for key, value in res.items()})
        row.update(cols)
        return pd.DataFrame([row])

    @staticmethod
    def _slugify(value):
        text = str(value).strip().lower()
        return "".join(c if (c.isalnum() or c in "-_.") else "-" for c in text)

    def _dataset_tag(self):
        parts = [self._slugify(self.cfg.dataset.name)]
        graph_type = self.cfg.dataset.get("graph_type", None)
        if graph_type:
            suffix = "_dag" if self.cfg.dataset.get("acyclic", False) else ""
            parts.append(self._slugify(graph_type) + suffix)
        return "-".join(parts)

    def _search_version_dir(self, search_name, tags=()):
        """Claim outputs/<search>_<dataset>_<tags>/version_N, resuming the latest unfinished one."""
        parts = [search_name, self._dataset_tag()] + [
            self._slugify(t) for t in tags if t is not None and str(t) != ""
        ]
        base_dir = os.path.abspath(
            os.path.join(get_original_cwd(), "..", "outputs", "_".join(parts))
        )
        os.makedirs(base_dir, exist_ok=True)

        for _ in range(100):
            versions = sorted(
                int(d.split("_")[1])
                for d in os.listdir(base_dir)
                if d.startswith("version_") and d.split("_")[1].isdigit()
            )
            if versions:
                latest_dir = os.path.join(base_dir, f"version_{versions[-1]}")
                if not os.path.exists(os.path.join(latest_dir, "DONE")):
                    print(f"Resuming search in {latest_dir}")
                    self._record_hydra_run(latest_dir)
                    return latest_dir
                next_version = versions[-1] + 1
            else:
                next_version = 0

            new_dir = os.path.join(base_dir, f"version_{next_version}")
            try:
                os.mkdir(new_dir)
            except FileExistsError:
                continue
            print(f"Starting search in {new_dir}")
            self._record_hydra_run(new_dir)
            return new_dir

        raise RuntimeError(
            f"Could not claim a version directory under {base_dir} after 100 "
            f"attempts -- too many jobs starting at once?"
        )

    def _axis_search_dir(self, search_name):
        """Flat outputs/<dataset>-<search_name>, reused across runs."""
        out_dir = os.path.abspath(
            os.path.join(
                get_original_cwd(),
                "..",
                "outputs",
                f"{self._dataset_tag()}-{search_name}",
            )
        )
        os.makedirs(out_dir, exist_ok=True)
        print(f"Writing {search_name} results to {out_dir}")
        self._record_hydra_run(out_dir)
        return out_dir

    def _record_hydra_run(self, version_dir):
        try:
            from hydra.core.hydra_config import HydraConfig

            run_dir = os.path.abspath(HydraConfig.get().runtime.output_dir)
        except Exception:
            run_dir = os.getcwd()  # not in a Hydra context (e.g. a unit test)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S %Z")
        with open(os.path.join(version_dir, "hydra_runs.txt"), "a") as f:
            f.write(f"{stamp}\t{run_dir}\n")

    def _mark_search_done(self, version_dir):
        open(os.path.join(version_dir, "DONE"), "w").close()

    def _load_search_checkpoint(self, csv_path, key_cols, dtypes):
        """Previously completed rows, plus the set of config tuples to skip."""
        if not os.path.exists(csv_path):
            return pd.DataFrame(), set()
        existing = pd.read_csv(csv_path, float_precision="round_trip")
        existing = existing.loc[:, ~existing.columns.str.match(r"^(Unnamed|index$|level_\d+$)")]
        for col, caster in dtypes.items():
            existing[col] = existing[col].apply(caster)
        completed = set(existing[key_cols].apply(tuple, axis=1))
        print(
            f"Resuming from checkpoint {csv_path}: "
            f"{len(completed)} combo(s) already completed."
        )
        return existing, completed

    def _save_search_checkpoint(self, results_df, csv_path):
        tmp_path = f"{csv_path}.tmp"
        results_df.to_csv(tmp_path)
        os.replace(tmp_path, csv_path)  # never leave a half-written checkpoint

    def _apply_sampling_config(self, num_step=None, distortor=None, eta=None, omega=None, a=None, b=None):
        if num_step is not None:
            self.cfg.sample.sample_steps = num_step
        if distortor is not None:
            self.cfg.sample.time_distortion = distortor
        if eta is not None:
            self.cfg.sample.eta = self.rate_matrix_designer.eta = eta
        if omega is not None:
            self.cfg.sample.omega = self.rate_matrix_designer.omega = omega
        if a is not None:
            self.cfg.sample.distortion_a = self.time_distorter.distortion_a = a
        if b is not None:
            self.cfg.sample.distortion_b = self.time_distorter.distortion_b = b

    def _reset_sampling_config(self):
        self._apply_sampling_config(distortor="identity", eta=0.0, omega=0.0, a=1.0, b=1.0)

    def _print_progress(self, config_time, done, total, search_start):
        elapsed = time.time() - search_start
        remaining = elapsed / done * max(total - done, 0)
        print(
            f"  -> took {config_time:.2f}s | elapsed: {elapsed:.2f}s "
            f"({elapsed / 60:.2f} min) | ETA remaining: {remaining:.2f}s "
            f"({remaining / 60:.2f} min)"
        )

    def _omega_power_transform(self, omega_linear):
        """omega = lo + (hi-lo)*(1-u**root); root < 1 concentrates the draws near lo."""
        omega_low, omega_high = self.cfg.sample.search_omega_range
        span = omega_high - omega_low
        # root == 1 degenerates to the mirror u -> 1-u, so pass the draw through
        if span == 0 or self.cfg.sample.search_random_omega_root == 1.0:
            return omega_linear
        u = (omega_linear - omega_low) / span
        return omega_low + (1.0 - u ** self.cfg.sample.search_random_omega_root) * span

    def _ask(self, study, search_space):
        # GPSampler optimizes its acquisition function with autograd, which the
        # inference_mode trainer.test() runs under would otherwise block
        with torch.inference_mode(False), torch.enable_grad():
            return study.ask(search_space)

    def _replay_trials(self, study, results_df, num_step, search_space, param_cols, objective_cols, label):
        """Re-ask every recorded trial and tell it its recorded value."""
        import optuna

        if not len(results_df):
            return 0
        prior = results_df[results_df["num_step"] == num_step].sort_values("trial_idx")

        for replay_idx, (_, row) in enumerate(prior.iterrows()):
            trial = self._ask(study, search_space)
            mismatched = [
                f"{param}: recorded {row[col]!r}, replayed {trial.params[param]!r}"
                for param, col in param_cols.items()
                if not (
                    np.isclose(float(trial.params[param]), float(row[col]), rtol=1e-9, atol=1e-12)
                    if isinstance(trial.params[param], float)
                    else str(trial.params[param]) == str(row[col])
                )
            ]
            if mismatched:
                raise RuntimeError(
                    f"{label} resume: replayed trial {replay_idx} (num_step={num_step}) did not "
                    f"reproduce the recorded proposal -- " + "; ".join(mismatched) + ".\n"
                    f"The sampler state could not be reconstructed, so this would NOT continue "
                    f"the original search. Check that search_seed, search_eta_range, "
                    f"search_omega_range, the sampler settings and the optuna version all match "
                    f"the run being resumed."
                )
            values = [float(row[col]) for col in objective_cols]
            if all(np.isfinite(v) for v in values):
                study.tell(trial, values[0] if len(values) == 1 else values)
            else:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)

        if len(prior):
            print(
                f"Replayed {len(prior)} trial(s) for num_step={num_step} into the {label} "
                f"study: every proposal was regenerated and verified, so the sampler state is "
                f"exactly reconstructed and no model inference was re-run."
            )
        return len(prior)

    def _make_bo_sampler(self, sampler_name, n_startup_trials):
        import optuna

        seed = self.cfg.sample.search_seed
        if sampler_name == "tpe":
            return optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials)
        if sampler_name == "gp":
            return optuna.samplers.GPSampler(seed=seed, n_startup_trials=n_startup_trials)
        raise ValueError(f"Unknown search_bo_sampler '{sampler_name}'. Choose 'tpe' or 'gp'.")

    def _bo_distortion_space(self, mode):
        """The distortion dimension(s) of the BO search space."""
        import optuna

        if mode == "continuous":
            a_low, a_high = self.cfg.sample.search_distortion_a_range
            b_low, b_high = self.cfg.sample.search_distortion_b_range
            return {
                "distortion_a": optuna.distributions.FloatDistribution(a_low, a_high),
                "distortion_b": optuna.distributions.FloatDistribution(b_low, b_high),
            }
        if mode == "categorical":
            return {
                "time_distortion": optuna.distributions.CategoricalDistribution(
                    ["identity", "polydec", "cos", "revcos", "polyinc"]
                )
            }
        raise ValueError(
            f"Unknown search_bo_distortion_mode '{mode}'. Choose 'continuous' or 'categorical'."
        )

    def _bo_apply_distortion(self, params, mode):
        """Push this trial's distortion live; return (label, {csv column: value})."""
        if mode == "continuous":
            a, b = float(params["distortion_a"]), float(params["distortion_b"])
            self._apply_sampling_config(distortor="continuous", a=a, b=b)
            return f"a: {a:.4f}, b: {b:.4f}", {"distortion_a": a, "distortion_b": b}
        distortor = str(params["time_distortion"])
        self._apply_sampling_config(distortor=distortor)
        return f"distortor: {distortor}", {"distortor": distortor}

    def _save_optuna_visualizations(self, study, num_step, sampler_name):
        from optuna import visualization as viz

        if not viz.is_available():
            print("  [viz] plotly not installed; skipping Optuna plots")
            return
        tag = f"{sampler_name}_numstep{num_step}"
        plots = (
            (("pareto_front", viz.plot_pareto_front), ("slice", viz.plot_slice))
            if len(study.directions) > 1
            else (
                ("optimization_history", viz.plot_optimization_history),
                ("slice", viz.plot_slice),
                ("param_importances", viz.plot_param_importances),
            )
        )
        for name, build in plots:
            try:
                build(study).write_html(f"optuna_{tag}_{name}.html")
            except Exception as e:
                print(f"  [viz] skip {name} for {tag}: {e}")
        print(f"  [viz] Optuna plots saved for {tag} (optuna_{tag}_*.html)")

    def _write_search_summary(self, search_times=None):
        with open("search_summary.txt", "w") as f:
            f.write(f"search: {self.cfg.sample.search}\n")
            f.write(f"status: {'completed' if search_times else 'running'}\n")
            f.write(f"started: {self._search_started_at}\n")
            for line in self._search_summary_info:
                f.write(f"{line}\n")
            if search_times:
                f.write("\nTime this run:\n")
                for name, t in search_times.items():
                    f.write(f"{name}: {t:.2f}s ({t / 60:.2f} min)\n")

    FIXED_CONFIG_HEADER = ["method", "objective", "time", "a", "b", "eta", "omega"]
    FIXED_CONFIG_NUMERIC = {"eta": 0.0, "omega": 0.0, "a": 1.0, "b": 1.0}

    @staticmethod
    def _read_fixed_configs_csv(csv_path):
        """Rows are allowed to omit one of the leading descriptive fields (the
        vanilla row carries no 'objective'), so short rows are padded there
        instead of at the end where the numbers live."""
        with open(csv_path, newline="") as f:
            rows = [
                [cell.strip() for cell in row]
                for row in csv.reader(f)
                if any(cell.strip() for cell in row)
            ]
        if not rows:
            raise ValueError(f"sample.search_configs_csv: '{csv_path}' is empty.")

        header, records = rows[0], []
        for line_no, row in enumerate(rows[1:], start=2):
            if len(row) < len(header):
                print(
                    f"  [fixed_configs] line {line_no} of {os.path.basename(csv_path)} has "
                    f"{len(row)} of {len(header)} fields, padding after '{header[0]}'"
                )
                row = row[:1] + [""] * (len(header) - len(row)) + row[1:]
            records.append(dict(zip(header, row[: len(header)])))
        return header, records

    def _load_fixed_configs(self):
        """The (distortor, a, b, eta, omega) list search 'fixed_configs' evaluates,
        read from sample.search_configs_csv."""
        csv_path = self.cfg.sample.search_configs_csv
        if csv_path is None:
            raise ValueError(
                "search: 'fixed_configs' needs sample.search_configs_csv to point at a "
                f"CSV with columns {self.FIXED_CONFIG_HEADER}."
            )
        csv_path = os.path.abspath(
            os.path.join(get_original_cwd(), os.path.expanduser(str(csv_path)))
        )
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"sample.search_configs_csv: no such file '{csv_path}'.")

        header, records = self._read_fixed_configs_csv(csv_path)
        if not records:
            raise ValueError(
                f"sample.search_configs_csv: '{csv_path}' holds a header but no config rows."
            )
        missing = [col for col in self.FIXED_CONFIG_HEADER if col not in header]
        if missing:
            raise KeyError(
                f"sample.search_configs_csv: '{csv_path}' is missing column(s) {missing}; "
                f"expected {self.FIXED_CONFIG_HEADER}, found {header}."
            )

        configs = []
        for rec in records:
            config = {"distortor": rec["time"]}
            for col, default in self.FIXED_CONFIG_NUMERIC.items():
                config[col] = float(rec[col]) if rec[col] != "" else default
            # every CSV column, in its original order, so the descriptive fields
            # (method, objective) reach the outputs
            config["row"] = dict(
                rec, **{col: config[col] for col in self.FIXED_CONFIG_NUMERIC}
            )
            configs.append(config)
        return configs, csv_path, header

    def _check_fixed_configs_snapshot(self, version_dir, csv_path):
        """The checkpoint keys its rows by config_idx, so a resumed run has to be
        reading the same config list: snapshot it the first time, compare after."""
        snapshot_path = os.path.join(version_dir, "configs.csv")
        with open(csv_path, newline="") as f:
            content = f.read()
        if not os.path.exists(snapshot_path):
            with open(snapshot_path, "w", newline="") as f:
                f.write(content)
            return
        with open(snapshot_path, newline="") as f:
            if f.read() != content:
                raise RuntimeError(
                    f"'{version_dir}' holds an unfinished 'fixed_configs' search over a "
                    f"different config list than '{csv_path}' (snapshot: {snapshot_path}). "
                    f"Its rows are keyed by config_idx, so resuming would mix the two. "
                    f"Mark that directory DONE (or delete it) to start a new version."
                )

    def _save_fixed_configs_seed_stats(self, results_df, configs_df, csv_header, stats_path):
        """mean/sd over the seeds, per config and num_step, with the source CSV's
        columns describing each config prepended."""
        # a molecular dataset has no ratio metric, so aggregate whichever of the
        # two objectives was actually evaluated
        aggs = {}
        for col, _ in zip(*objective_spec("both", self.cfg.dataset.directed)):
            if col in results_df.columns:
                name = col.split("/")[-1].removesuffix("_mean")
                aggs[f"{name}_mean"] = (col, "mean")
                aggs[f"{name}_sd"] = (col, "std")
        if not aggs:
            return
        seed_stats = results_df.groupby(["config_idx", "num_step"]).agg(**aggs).reset_index()
        seed_stats = configs_df.merge(seed_stats, on="config_idx")
        seed_stats = seed_stats[csv_header + ["config_idx", "num_step"] + list(aggs)]
        seed_stats.to_csv(stats_path, index=False)
