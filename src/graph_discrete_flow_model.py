import time
import wandb
import os
import random
import itertools

import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.distributions.categorical import Categorical
from hydra.utils import get_original_cwd

from models.transformer_model import GraphTransformer
from models.transformer_model_directed import GraphTransformerDirected

from metrics.train_metrics import TrainLossDiscrete
from src import utils
from flow_matching.noise_distribution import NoiseDistribution
from flow_matching.time_distorter import TimeDistorter
from flow_matching.rate_matrix import RateMatrixDesigner
from flow_matching.utils import p_xt_g_x1
from flow_matching import flow_matching_utils


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


class GraphDiscreteFlowModel(pl.LightningModule):
    def __init__(
        self,
        cfg,
        dataset_infos,
        train_metrics,
        sampling_metrics,
        visualization_tools,
        extra_features,
        domain_features,
        test_labels=None,
    ):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = f"{cfg.dataset.name}_{cfg.general.name}"
        self.model_dtype = torch.float32
        self.conditional = cfg.general.conditional
        self.test_labels = test_labels

        # number of steps used for sampling
        self.sample_T = cfg.sample.sample_steps

        self.input_dims = dataset_infos.input_dims
        self.output_dims = dataset_infos.output_dims
        self.dataset_info = dataset_infos
        self.directed = cfg.dataset.directed
        self.node_dist = dataset_infos.nodes_dist
        print("max num nodes: ", len(self.node_dist.prob) - 1)
        print("min num nodes: ", torch.where(self.node_dist.prob > 0)[0][0])

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.noise_dist = NoiseDistribution(cfg.model.transition, dataset_infos)
        self.limit_dist = self.noise_dist.get_limit_dist()

        # add virtual class when absorbing state refers to a new class
        self.noise_dist.update_input_output_dims(self.input_dims)
        self.noise_dist.update_dataset_infos(self.dataset_info)

        self.train_loss = TrainLossDiscrete(
            self.cfg.model.lambda_train,
        )

        if cfg.dataset.directed:
            self.model = GraphTransformerDirected(
                n_layers=cfg.model.n_layers,
                directed=self.directed,
                pos_enc=cfg.model.pos_enc,
                input_dims=input_dims,
                hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                hidden_dims=cfg.model.hidden_dims,
                output_dims=output_dims,
                act_fn_in=nn.ReLU(),
                act_fn_out=nn.ReLU(),
                dual=cfg.model.dual,
                pos_enc_frac=cfg.model.pos_enc_frac,
                pos_enc_mode=cfg.model.pos_enc_mode,
            )
        else:
            self.model = GraphTransformer(
                n_layers=cfg.model.n_layers,
                input_dims=input_dims,
                hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                hidden_dims=cfg.model.hidden_dims,
                output_dims=output_dims,
                act_fn_in=nn.ReLU(),
                act_fn_out=nn.ReLU(),
            )

        self.save_hyperparameters(
            ignore=[
                "train_metrics",
                "sampling_metrics",
            ],
        )

        # logging
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.val_counter = 0
        self.adapt_counter = 0

        # time distortor for both training and sampling steps
        self.time_distorter = TimeDistorter(
            train_distortion=cfg.train.time_distortion,
            sample_distortion=cfg.sample.time_distortion,
            alpha=1,
            beta=1,
            distortion_a=cfg.sample.distortion_a,
            distortion_b=cfg.sample.distortion_b,
        )

        # rate matrix designer
        self.rate_matrix_designer = RateMatrixDesigner(
            rdb=self.cfg.sample.rdb,
            rdb_crit=self.cfg.sample.rdb_crit,
            eta=self.cfg.sample.eta,
            omega=self.cfg.sample.omega,
            limit_dist=self.limit_dist,
        )

    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            self.print("Found a batch with no edges. Skipping.")
            return
        
        if self.conditional:
            if torch.rand(1) < 0.1:
                data.y = torch.ones_like(data.y, device=self.device) * -1

        dense_data, node_mask = utils.to_dense(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        dense_data = dense_data.mask(node_mask, directed=self.directed)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        loss = self.train_loss(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            pred_y=pred.y,
            true_X=X,
            true_E=E,
            true_y=data.y,
            log=i % self.log_every_steps == 0,
        )

        self.train_metrics(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            true_X=X,
            true_E=E,
            log=i % self.log_every_steps == 0,
        )

        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.train.lr,
            amsgrad=True,
            weight_decay=self.cfg.train.weight_decay,
        )

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        self.print(
            "Size of the input features",
            self.input_dims["X"],
            self.input_dims["E"],
            self.input_dims["y"],
        )
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def on_train_epoch_start(self) -> None:
        self.print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        self.print(
            f"Epoch {self.current_epoch}: X_CE: {to_log['train_epoch/x_CE'] :.3f}"
            f" -- E_CE: {to_log['train_epoch/E_CE'] :.3f} --"
            f" y_CE: {to_log['train_epoch/y_CE'] :.3f}"
            f" -- {time.time() - self.start_epoch_time:.1f}s "
        )
        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        self.print(
            f"Epoch {self.current_epoch}: {epoch_at_metrics} -- {epoch_bond_metrics}"
        )
        if wandb.run:
            wandb.log({"epoch": self.current_epoch}, commit=False)

    def on_validation_epoch_start(self) -> None:
        print("Starting validation...")
        self.sampling_metrics.reset()

    def validation_step(self, data, i):
        return

    def on_validation_epoch_end(self) -> None:
        self.val_counter += 1
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            print("Starting to sample")
            samples, labels = self.sample(
                is_test=False, save_samples=False, save_visualization=False
            )
            to_log = self.evaluate_samples(
                samples=samples, labels=labels, is_test=False
            )

            # Store results
            filename = os.path.join(
                os.getcwd(),
                f"val_epoch{self.current_epoch}_res_{self.cfg.sample.eta}_{self.cfg.sample.rdb}.txt",
            )
            with open(filename, "w") as file:
                for key, value in to_log.items():
                    file.write(f"{key}: {value}\n")

        self.print("Finished validation.")

    def on_test_epoch_start(self) -> None:
        self.print("Starting test...")
        self.sampling_metrics.reset()
        if self.local_rank == 0:
            utils.setup_wandb(self.cfg)

    def test_step(self, data, i):
        return

    def on_test_epoch_end(self) -> None:
        print("sampling optimization")
        if self.cfg.sample.search:
            self.search_hyperparameters()
        else:
            print("Starting to sample")
            samples, labels = self.sample(
                is_test=True, save_samples=self.cfg.general.save_samples, save_visualization=self.cfg.general.save_visualization
            )
            to_log = self.evaluate_samples(samples=samples, labels=labels, is_test=True)

            # Store results
            filename = os.path.join(
                os.getcwd(),
                f"test_epoch{self.current_epoch}_res_{self.cfg.sample.eta}_{self.cfg.sample.rdb}.txt",
            )
            with open(filename, "w") as file:
                for key, value in to_log.items():
                    file.write(f"{key}: {value}\n")

            self.print("Finished testing.")

    def sample(self, is_test, save_samples, save_visualization):

        # Logic to load generated sampled instead of resampling in case of having them saved, leaving commented for now as backbone:diffusion does not have it
        # # Load generated samples if they exist
        # if self.cfg.general.generated_path:
        #     self.print("Loading generated samples...")
        #     with open(self.cfg.general.generated_path, "rb") as f:
        #         samples = pickle.load(f)
        #     # Set labels to None
        #     labels = [None] * len(samples)
        #     return samples, None

        # Otherwise, generate new samples
        if is_test:
            if self.cfg.general.bootstrapping and self.cfg.general.num_sample_fold != 1:
                raise ValueError(
                    "When bootstrapping is enabled, num_sample_fold must be 1."
                )
            samples_to_generate = (
                self.cfg.general.final_model_samples_to_generate
                * self.cfg.general.num_sample_fold
            )
            samples_left_to_generate = (
                self.cfg.general.final_model_samples_to_generate
                * self.cfg.general.num_sample_fold
            )
            samples_left_to_save = self.cfg.general.final_model_samples_to_save
            chains_left_to_save = self.cfg.general.final_model_chains_to_save

        else:
            samples_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save

        samples = []
        labels = []
        graph_id = 0
        while samples_left_to_generate > 0:
            self.print(
                f"Samples left to generate: {samples_left_to_generate}/"
                f"{samples_to_generate}",
                end="",
                # flush=True,
            )
            bs = 2 * self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            num_chain_steps = min(self.number_chain_steps, self.sample_T)
            cur_samples, cur_labels = self.sample_batch(
                graph_id,
                to_generate,
                num_nodes=None,
                save_final=to_save,
                keep_chain=chains_save,
                number_chain_steps=num_chain_steps,
                save_visualization=save_visualization,
                test=is_test
            )
            samples.extend(cur_samples)
            labels.extend(cur_labels)

            graph_id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save

        if save_samples:
            self.print("Saving the generated graphs")

            # saving in txt version
            filename = "graphs.txt"
            with open(filename, "w") as f:
                for item in samples:
                    f.write(f"N={item[0].shape[0]}\n")
                    atoms = item[0].tolist()
                    f.write("X: \n")
                    for at in atoms:
                        f.write(f"{at} ")
                    f.write("\n")
                    f.write("E: \n")
                    for bond_list in item[1]:
                        for bond in bond_list:
                            f.write(f"{bond} ")
                        f.write("\n")
                    f.write("\n")

            # saving in pkl version
            with open(f"generated_samples_rank{self.local_rank}.pkl", "wb") as f:
                pickle.dump(samples, f)

            print("Generated graphs saved.")

        return samples, labels

    def evaluate_samples(
        self,
        samples,
        labels,
        is_test,
        save_filename="",
    ):
        print("Computing sampling metrics...")

        to_log = {}
        samples_to_evaluate = self.cfg.general.final_model_samples_to_generate
        if is_test:
            if self.cfg.general.bootstrapping:
                num_bootstrap_fold = self.cfg.general.num_bootstrap_fold
                n_total = len(samples)
                # each fold scores (K-1)/K of the generated pool
                n_samples_to_evaluate = max(
                    1, (n_total * (num_bootstrap_fold - 1)) // num_bootstrap_fold
                )

                fold_indices = [
                    np.random.choice(n_total, size=n_samples_to_evaluate, replace=False)
                    for _ in range(num_bootstrap_fold)
                ]
            else:
                fold_indices = [
                    range(i * samples_to_evaluate, (i + 1) * samples_to_evaluate)
                    for i in range(self.cfg.general.num_sample_fold)
                ]

            for i, idx in enumerate(fold_indices):
                cur_samples = [samples[j] for j in idx]
                cur_labels = [labels[j] for j in idx]

                cur_to_log = self.sampling_metrics.forward(
                    cur_samples,
                    ref_metrics=self.dataset_info.ref_metrics,
                    name=f"self.name_{i}",
                    current_epoch=self.current_epoch,
                    val_counter=-1,
                    test=is_test,
                    local_rank=self.local_rank,
                )

                if i == 0:
                    to_log = {i: [cur_to_log[i]] for i in cur_to_log}
                else:
                    to_log = {i: to_log[i] + [cur_to_log[i]] for i in cur_to_log}

                filename = os.path.join(
                    os.getcwd(),
                    f"epoch{self.current_epoch}_res_fold{i}_{save_filename}.txt",
                )
                with open(filename, "w") as file:
                    for key, value in cur_to_log.items():
                        file.write(f"{key}: {value}\n")

            to_log = {
                i: (np.array(to_log[i]).mean(), np.array(to_log[i]).std())
                for i in to_log
            }
        else:
            to_log = self.sampling_metrics.forward(
                samples,
                ref_metrics=self.dataset_info.ref_metrics,
                name=self.cfg.general.name,
                current_epoch=self.current_epoch,
                val_counter=-1,
                test=is_test,
                local_rank=self.local_rank,
            )

        return to_log

    def apply_noise(self, X, E, y, node_mask, t=None):
        """Sample noise and apply it to the data."""

        # Sample a timestep t.
        bs = X.size(0)
        if t is None:
            t_float = self.time_distorter.train_ft(bs, self.device)
        else:
            t_float = t

        # sample random step
        X_1_label = torch.argmax(X, dim=-1)
        E_1_label = torch.argmax(E, dim=-1)
        prob_X_t, prob_E_t = p_xt_g_x1(
            X1=X_1_label, E1=E_1_label, t=t_float, limit_dist=self.limit_dist
        )

        # step 4 - sample noised data
        sampled_t = flow_matching_utils.sample_discrete_features(
            probX=prob_X_t, probE=prob_E_t, node_mask=node_mask
        )
        noise_dims = self.noise_dist.get_noise_dims()
        X_t = F.one_hot(sampled_t.X, num_classes=noise_dims["X"])
        E_t = F.one_hot(sampled_t.E, num_classes=noise_dims["E"])

        # step 5 - create the PlaceHolder
        z_t = (
            utils.PlaceHolder(X=X_t, E=E_t, y=y)
            .type_as(X_t)
            .mask(node_mask, directed=self.directed)
        )

        noisy_data = {
            "t": t_float,
            "X_t": z_t.X,
            "E_t": z_t.E,
            "y_t": z_t.y,
            "node_mask": node_mask,
        }

        return noisy_data

    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data["X_t"], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data["E_t"], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data["y_t"], extra_data.y)).float()
        return self.model(X, E, y, node_mask)

    @torch.no_grad()
    def sample_batch(
        self,
        batch_id: int,
        batch_size: int,
        keep_chain: int,
        number_chain_steps: int,
        save_final: int,
        num_nodes=None,
        save_visualization: bool = False,
        test: bool = False
    ):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(
                batch_size, device=self.device, dtype=torch.int
            )
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()

        # Build the masks
        arange = (
            torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        )
        node_mask = arange < n_nodes.unsqueeze(1)

        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = flow_matching_utils.sample_discrete_feature_noise(
            limit_dist=self.noise_dist.get_limit_dist(),
            node_mask=node_mask,
            directed=self.directed,
        )
        if self.conditional:
            if "qm9" in self.cfg.dataset.name:
                y = self.test_labels
                perm = torch.randperm(y.size(0))
                idx = perm[:100]
                condition = y[idx]
                condition = condition.to(self.device)
                z_T.y = condition.repeat([10, 1])[:batch_size, :]
            elif "tls" in self.cfg.dataset.name:
                z_T.y = torch.zeros(batch_size, 1).to(self.device)
                z_T.y[: batch_size // 2] = 1
            elif "tpu_tile" in self.cfg.dataset.name:
                y = self.test_labels
                counts, bin_edges = torch.histogram(y)
                probs = counts.float() / counts.sum()
                distribution = torch.distributions.Categorical(probs)

                if test:
                    # For sampling
                    condition = y[batch_id : (batch_id + batch_size)]
                    condition = condition.to(self.device)
                else:
                    # For training
                    sample = distribution.sample((batch_size,))
                    bin_left = bin_edges[sample]
                    bin_right = bin_edges[sample + 1]
                    condition = torch.randn_like(bin_left) * (bin_right - bin_left) + bin_left
                    condition = condition.unsqueeze(1).to(self.device)

                z_T.y = condition

            else:
                raise NotImplementedError
        X, E, y = z_T.X, z_T.E, z_T.y

        # Init chain storing variables
        # assert (E == torch.transpose(E, 1, 2)).all() # Remove symmetrization assert
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size(
            (number_chain_steps, keep_chain, E.size(1), E.size(2))
        )
        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)
        chain_times = torch.zeros((number_chain_steps, keep_chain))
        chain_time_unit = 1 / (number_chain_steps - 1)

        # Store initial graph
        if keep_chain > 0:
            sampled_initial = z_T.mask(node_mask, collapse=True, directed=self.directed)
            chain_X[0] = sampled_initial.X[:keep_chain]
            chain_E[0] = sampled_initial.E[:keep_chain]
            chain_times[0] = torch.zeros((keep_chain))

        for t_int in tqdm(range(0, self.cfg.sample.sample_steps)):
            # this state
            t_array = t_int * torch.ones((batch_size, 1)).type_as(y)
            t_norm = t_array / self.cfg.sample.sample_steps
            if ("absorb" in self.cfg.model.transition) and (t_int == 0):
                # to avoid failure mode of absorbing transition, add epsilon
                t_norm = t_norm + 1e-6
            # next state
            s_array = t_array + 1
            s_norm = s_array / self.cfg.sample.sample_steps

            # Distort time
            t_norm = self.time_distorter.sample_ft(
                t_norm, self.cfg.sample.time_distortion
            )
            s_norm = self.time_distorter.sample_ft(
                s_norm, self.cfg.sample.time_distortion
            )

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(
                t_norm,
                s_norm,
                X,
                E,
                y,
                node_mask,
            )

            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # Save the first keep_chain graphs
            # using round for precision
            write_index = int(np.ceil(np.round(s_norm[0].item() / chain_time_unit, 6)))
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]
            chain_times[write_index] = s_norm.flatten()[:keep_chain]

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True, directed=self.directed)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Prepare the chain for saving
        if keep_chain > 0:

            # Repeat last frame 10x to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            chain_times = torch.cat(
                [chain_times, chain_times[-1:].repeat(10, 1)], dim=0
            )
            assert chain_X.size(0) == (number_chain_steps + 10)

        X, E, y = self.noise_dist.ignore_virtual_classes(X, E, y)
        chain_X, chain_E, _ = self.noise_dist.ignore_virtual_classes(
            chain_X, chain_E, y
        )

        # Save generated graphs
        molecule_list = []
        label_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
            label_list.append(y[i].cpu())

        if self.visualization_tools is not None and save_visualization:
            # Visualize chains
            self.print("Visualizing chains...")
            current_path = os.getcwd()
            num_molecules = chain_X.size(1)  # number of molecules
            for i in range(num_molecules):
                result_path = os.path.join(
                    current_path,
                    f"chains/{self.cfg.general.name}/"
                    f"epoch{self.current_epoch}/"
                    f"chains/molecule_{batch_id + i}",
                )
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    _ = self.visualization_tools.visualize_chain(
                        result_path,
                        chain_X[:, i, :].numpy(),
                        chain_E[:, i, :].numpy(),
                        # chain_times[:, i].numpy(),
                    )
                self.print(
                    "\r{}/{} complete".format(i + 1, num_molecules),
                    end="",
                    # flush=True
                )
            self.print("\nVisualizing graphs...")

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(
                current_path,
                f"graphs/{self.cfg.general.name}/epoch{self.current_epoch}_b{batch_id}/",
            )
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            self.print("Done.")

        return molecule_list, label_list

    def compute_step_probs(self, R_t_X, R_t_E, X_t, E_t, dt, limit_x, limit_e):
        step_probs_X = R_t_X * dt  # type: ignore # (B, D, S)
        step_probs_E = R_t_E * dt  # (B, D, S)

        # Calculate the on-diagnoal step probabilities
        # 1) Zero out the diagonal entries
        # assert (E_t.argmax(-1) < 4).all()
        step_probs_X.scatter_(-1, X_t.argmax(-1)[:, :, None], 0.0)
        step_probs_E.scatter_(-1, E_t.argmax(-1)[:, :, :, None], 0.0)

        # 2) Calculate the diagonal entries such that the probability row sums to 1
        step_probs_X.scatter_(
            -1,
            X_t.argmax(-1)[:, :, None],
            (1.0 - step_probs_X.sum(dim=-1, keepdim=True)).clamp(min=0.0),
        )
        step_probs_E.scatter_(
            -1,
            E_t.argmax(-1)[:, :, :, None],
            (1.0 - step_probs_E.sum(dim=-1, keepdim=True)).clamp(min=0.0),
        )

        # step 2 - merge to the original formulation
        prob_X = step_probs_X.clone()
        prob_E = step_probs_E.clone()

        return prob_X, prob_E

    def sample_p_zs_given_zt(
        self,
        t,
        s,
        X_t,
        E_t,
        y_t,
        node_mask,
        # , condition
    ):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
        if last_step, return the graph prediction as well"""
        bs, n, dx = X_t.shape
        _, _, _, de = E_t.shape
        dt = (s - t)[0]

        # Neural net predictions
        noisy_data = {
            "X_t": X_t,
            "E_t": E_t,
            "y_t": y_t,
            "t": t,
            "node_mask": node_mask,
        }

        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0
        limit_x = self.limit_dist.X
        limit_e = self.limit_dist.E

        G_1_pred = pred_X, pred_E
        G_t = X_t, E_t

        R_t_X, R_t_E = self.rate_matrix_designer.compute_graph_rate_matrix(
            t,
            node_mask,
            G_t,
            G_1_pred,
        )

        if self.conditional:
            uncond_y = torch.ones_like(y_t, device=self.device) * -1
            noisy_data["y_t"] = uncond_y

            extra_data = self.compute_extra_data(noisy_data)
            pred = self.forward(noisy_data, extra_data, node_mask)

            pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
            pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0

            R_t_X_uncond, R_t_E_uncond = (
                self.rate_matrix_designer.compute_graph_rate_matrix(
                    t,
                    node_mask,
                    G_t,
                    G_1_pred,
                )
            )

            guidance_weight = self.cfg.general.guidance_weight
            R_t_X = torch.exp(
                torch.log(R_t_X_uncond + 1e-6) * (1 - guidance_weight)
                + torch.log(R_t_X + 1e-6) * guidance_weight
            )
            R_t_E = torch.exp(
                torch.log(R_t_E_uncond + 1e-6) * (1 - guidance_weight)
                + torch.log(R_t_E + 1e-6) * guidance_weight
            )

        prob_X, prob_E = self.compute_step_probs(
            R_t_X, R_t_E, X_t, E_t, dt, limit_x, limit_e
        )

        if s[0] == 1.0:
            prob_X, prob_E = pred_X, pred_E

        sampled_s = flow_matching_utils.sample_discrete_features(
            prob_X, prob_E, node_mask=node_mask
        )

        X_s = F.one_hot(sampled_s.X, num_classes=len(limit_x)).float()
        E_s = F.one_hot(sampled_s.E, num_classes=len(limit_e)).float()

        # assert (E_s == torch.transpose(E_s, 1, 2)).all() # Remove symmetrization assert
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        if self.conditional:
            y_to_save = y_t
        else:
            y_to_save = torch.zeros([y_t.shape[0], 0], device=self.device)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=y_to_save)
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=y_to_save)

        out_one_hot = out_one_hot.mask(node_mask, directed=self.directed).type_as(y_t)
        out_discrete = out_discrete.mask(
            node_mask, collapse=True, directed=self.directed
        ).type_as(y_t)

        return out_one_hot, out_discrete

    def compute_extra_data(self, noisy_data):
        """At every training step (after adding noise) and step in sampling, compute extra information and append to
        the network input."""

        extra_features = self.extra_features(noisy_data)

        # one additional category is added for the absorbing transition
        X, E, y = self.noise_dist.ignore_virtual_classes(
            noisy_data["X_t"], noisy_data["E_t"], noisy_data["y_t"]
        )
        noisy_data_to_mol_feat = noisy_data.copy()
        noisy_data_to_mol_feat["X_t"] = X
        noisy_data_to_mol_feat["E_t"] = E
        noisy_data_to_mol_feat["y_t"] = y
        extra_molecular_features = self.domain_features(noisy_data_to_mol_feat)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data["t"]
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

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

    def _search_version_dir(self, search_name, tags=()):
        """Claim outputs/<search>_<tags>/version_N, resuming the latest unfinished one."""
        parts = [search_name] + [
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

    def search_hyperparameters(self):
        """
        Grid search for sampling hypeparameters.
        The num_step_list is tunable based on requirements.
        """

        num_step_list = [50] #[50, 1000]  # [5, 10, 50, 100, 1000]

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
        """
        Grid search for sampling distortion.
        """
        results_df = pd.DataFrame()
        distortion_list = ["identity", "polydec", "cos", "revcos", "polyinc"]
        # distortion_list = ["identity", "polydec"]
        seed_list = [0, 1, 2]

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
                    # save at each step as well
                    results_df.to_csv(f"search_distortion.csv")

        # set back to default value
        self.cfg.sample.time_distortion = "identity"

        # save the final results
        results_df.reset_index(drop=True, inplace=True)
        results_df.set_index(["num_step", "distortor", "seed"], inplace=True)
        results_df.to_csv(f"search_distortion.csv")

    def search_stochasticity(self, num_step_list):
        """
        Grid search for stochasticity level eta.
        The num_step_list is tunable based on requirements.
        """
        results_df = pd.DataFrame()
        eta_list = [0.0, 5, 10, 25, 50, 100, 200]
        # eta_list = [5, 10]
        seed_list = [0, 1, 2]

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
                    # save at each step as well
                    results_df.to_csv(f"search_stochasticity.csv")

        self.cfg.sample.eta = 0.0
        self.rate_matrix_designer.eta = 0.0

        results_df.reset_index(drop=True, inplace=True)
        results_df.set_index(["num_step", "eta", "seed"], inplace=True)
        results_df.to_csv(f"search_stochasticity.csv")

    def search_target_guidance(self, num_step_list):
        """
        Grid search for target guidance omega.
        The num_step_list is tunable based on requirements.
        """
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
                    results_df.to_csv(f"search_target_guidance.csv")

        self.cfg.sample.omega = 0.0
        self.rate_matrix_designer.omega = 0.0

        results_df.reset_index(drop=True, inplace=True)
        results_df.set_index(["num_step", "omega", "seed"], inplace=True)
        results_df.to_csv(f"search_target_guidance.csv")

    def search_full_grid(self, num_step_list):
        """
        Grid search over distortion x eta x omega, checkpointed after every config.
        """
        distortion_list = ["identity", "polydec", "cos", "revcos", "polyinc"]
        eta_list = [0.0, 5, 10, 25, 50, 100]
        omega_list = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

        version_dir = self._search_version_dir("full_grid", tags=(self.cfg.dataset.name,))
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
        """
        Random search over distortion x eta x omega, checkpointed after every trial.
        """
        distortion_list = ["identity", "polydec", "cos", "revcos", "polyinc"]
        eta_low, eta_high = self.cfg.sample.search_eta_range
        omega_low, omega_high = self.cfg.sample.search_omega_range
        n_trials = self.cfg.sample.search_n_trials

        version_dir = self._search_version_dir(
            "random", tags=(self.cfg.dataset.name, f"seed{self.cfg.sample.search_seed}")
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
        """
        Sobol (scrambled QMC) search over distortion x eta x omega."""
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
            "sobol", tags=(self.cfg.dataset.name, f"seed{self.cfg.sample.search_seed}")
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
            tags=(self.cfg.dataset.name, sampler_name, mode, f"seed{self.cfg.sample.search_seed}"),
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
