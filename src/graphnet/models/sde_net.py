from graphnet.models.model import Model
from typing import List, TYPE_CHECKING, Any, Union

from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from torch_geometric.utils import to_dense_batch
from torch_geometric.data import Data
from torch import nn
from torch.optim import SGD
from copy import deepcopy
import torch
from torch import Tensor
import math
from icecream import ic
from graphnet.models.task import Task

if TYPE_CHECKING:
    from graphnet.models.gnn import GNN


class SDENet(Model):
    def __init__(
            self,
            encoder: "GNN",
            decoder: "GNN",
            drift: "GNN",
            diffusion: "GNN",
            tasks: Union[Task, List[Task]],
            total_time: float,
            delta_t: float,
            sigma_max: float = 50,
            latent_proj_dim: int = 128,
            ood_std_dev: float = 2.
    ):
        super().__init__(name=__name__, class_name=self.__class__.__name__)
        self._encoder = encoder
        self._decoder = decoder
        self._drift = drift
        self._diffusion = nn.Sequential(
            diffusion,
            nn.Linear(diffusion.nb_outputs, 1),
            nn.Sigmoid(),
        )
        if isinstance(tasks, Task):
            tasks = [tasks]
        self._tasks = nn.ModuleList(tasks)

        self._total_time = total_time
        self._delta_t = delta_t
        self._total_steps = int(total_time / delta_t)
        self.ood_std_dev = ood_std_dev

        self.sigma_max = sigma_max
        self._latent_proj_dim = latent_proj_dim
        self._time_graph_proj_drift = nn.Sequential(
            nn.Linear(self._drift.nb_outputs + 1, self._latent_proj_dim),
            nn.GELU(),
            nn.LayerNorm(self._latent_proj_dim),
            nn.Linear(self._latent_proj_dim, diffusion.nb_inputs)
        )
        self._time_graph_proj_diffusion = nn.Sequential(
            nn.Linear(self._drift.nb_outputs + 1, self._latent_proj_dim),
            nn.GELU(),
            nn.LayerNorm(self._latent_proj_dim),
            nn.Linear(self._latent_proj_dim, diffusion.nb_inputs)
        )
        self.ood_loss = nn.BCELoss()

        self.automatic_optimization = False

    def _verify_compatible_extractors(self):
        if self._encoder.nb_outputs != self._drift.nb_inputs:
            raise ValueError(
                "Output dimension of Encoder should match input dimension of drift."
            )
        if self._encoder.nb_outputs != self._diffusion.nb_inputs:
            raise ValueError(
                "Output dimension of Encoder should match input dimension of drift."
            )
        if self._drift.nb_inputs != self._drift.nb_outputs:
            raise ValueError(
                "Output and input dimension of drift moodel should be identical."
            )
        if self._diffusion.nb_outputs != 1:
            raise ValueError(
                "Diffusion should have output dimension 1."
            )

    def forward(self, data: Data, predict_if_ood: bool = False):
        encoded_graph = self._encoder(data)
        encoded_data = deepcopy(data)
        time_encoded_data = deepcopy(data)
        encoded_data.x = encoded_graph

        if predict_if_ood:
            encoded_graph.detach()
        t = 0
        graph_time = torch.ones_like(encoded_data.x[:, :1]) * t
        stacked_graphs = torch.cat([graph_time, encoded_data.x], dim=1)
        time_encoded_graph = self._time_graph_proj_diffusion(stacked_graphs)
        time_encoded_data.x = time_encoded_graph

        ood_prediction = self._diffusion(time_encoded_data)  # Same as diffusion fraction

        if predict_if_ood:
            return {"preds": None, "ood_preds": ood_prediction}

        diffusion = self.sigma_max * ood_prediction
        diffusion.unsqueeze_(2)
        dense_encoded_graph, dense_mask = to_dense_batch(encoded_data.x, encoded_data.batch)
        for i in range(self._total_steps):
            t = i * self._delta_t
            graph_time = torch.ones_like(encoded_data.x[:, :1]) * t
            stacked_graphs = torch.cat([graph_time, encoded_data.x], dim=1)
            time_encoded_graph = self._time_graph_proj_drift(stacked_graphs)
            diffusion_graph = (
                    diffusion
                    * math.sqrt(self._delta_t) * torch.randn_like(dense_encoded_graph)
            )[dense_mask]
            time_encoded_data.x = time_encoded_graph
            encoded_data.x = (
                    encoded_data.x
                    + self._drift(time_encoded_data) * self._delta_t
                    + diffusion_graph
            )
        x = self._decoder(encoded_data)
        preds = [task(x) for task in self._tasks]
        return {"preds": preds, "ood_preds": ood_prediction}

    def training_step(self, train_batch: Data, batch_idx: int):
        # In domain step
        opt_f, opt_g = self.optimizers()
        opt_f.zero_grad()
        preds = self(train_batch)
        loss = self._compute_loss(preds["preds"], train_batch)
        self.manual_backward(loss)
        opt_f.step()

        # Out of domain step
        label = torch.full((self._get_batch_size(train_batch), 1), 0., device=self.device, dtype=torch.float32)
        opt_g.zero_grad()
        preds_in = self(train_batch, predict_if_ood=True)
        loss_in_distribution = self.ood_loss(preds_in["ood_preds"], label)
        self.manual_backward(loss_in_distribution)

        label = torch.full((self._get_batch_size(train_batch), 1), 1.,
                           device=self.device, dtype=torch.float32)
        ood_train_batch = deepcopy(train_batch)
        ood_train_batch.x[:, [3, 4]] += self.ood_std_dev * torch.randn_like(ood_train_batch.x[:, [3, 4]])
        preds_out = self(ood_train_batch, predict_if_ood=True)
        loss_out_distribution = self.ood_loss(preds_out["ood_preds"], label)
        self.manual_backward(loss_out_distribution)

        opt_g.step()

        self.log(
            "train_f_loss",
            loss,
            batch_size=self._get_batch_size(train_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "train_g_loss_in",
            loss_out_distribution,
            batch_size=self._get_batch_size(train_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "train_g_loss_out",
            loss_out_distribution,
            batch_size=self._get_batch_size(train_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

    def validation_step(self, val_batch: Data, batch_idx: int):
        # In domain step
        preds = self(val_batch)
        loss = self._compute_loss(preds["preds"], val_batch)

        # Out of domain step
        label = torch.full(size=(self._get_batch_size(val_batch), 1), fill_value=0., device=self.device, dtype=torch.float32)
        preds_in = self(val_batch, predict_if_ood=True)
        loss_in_distribution = self.ood_loss(preds_in["ood_preds"], label)

        label.fill_(1)
        ood_val_batch = deepcopy(val_batch)
        ood_val_batch.x[:, [3, 4]] += self.ood_std_dev * torch.randn_like(ood_val_batch.x[:, [3, 4]])
        preds_out = self(ood_val_batch, predict_if_ood=True)
        loss_out_distribution = self.ood_loss(preds_out["ood_preds"], label)

        self.log(
            "val_f_loss",
            loss,
            batch_size=self._get_batch_size(val_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "val_g_loss_in",
            loss_out_distribution,
            batch_size=self._get_batch_size(val_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self.log(
            "val_g_loss_out",
            loss_out_distribution,
            batch_size=self._get_batch_size(val_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )

    def configure_optimizers(self):
        opt_f = SGD(
            [
                {"params": self._encoder.parameters()},
                {"params": self._drift.parameters()},
                {"params": self._time_graph_proj_drift.parameters()},
                {"params": self._decoder.parameters()},
                {"params": self._tasks.parameters()},
            ],
            lr=0.1, momentum=0.9, weight_decay=5e-4
        )
        opt_g = SGD(
            [
                {"params": self._diffusion.parameters()},
                {"params": self._time_graph_proj_diffusion.parameters()},
            ],
            lr=0.01, momentum=0.9, weight_decay=5e-4
        )
        return opt_f, opt_g

    def _compute_loss(
            self, preds: Tensor, data: Data, verbose: bool = False
    ) -> Tensor:
        """Compute and sum losses across tasks."""
        losses = [
            task.compute_loss(pred, data)
            for task, pred in zip(self._tasks, preds)
        ]
        if verbose:
            self.info(f"{losses}")
        assert all(
            loss.dim() == 0 for loss in losses
        ), "Please reduce loss for each task separately"
        return torch.mean(torch.stack(losses))


if __name__ == "__main__":
    from graphnet.models.gnn import DynEdgeTITO, DynEdge
    from graphnet.models.bept import DeepIceModelNoExtractor
    from torch_geometric.data import Batch
    from graphnet.debug_dev.dummy_data import DummyDataGenerator

    inputs = 7
    encoder = DynEdge(
        nb_inputs=inputs,
    )
    diffusion = DynEdge(
        nb_inputs=encoder.nb_outputs,
        dynedge_layer_sizes=[(512, 256)],
        global_pooling_schemes=["max"],
    )
    drift = DynEdge(
        nb_inputs=encoder.nb_outputs,
        dynedge_layer_sizes=[(512, 256), (512, 256)]
    )
    decoder = DynEdge(
        nb_inputs=encoder.nb_outputs,
        dynedge_layer_sizes=[(512, 256)],
        global_pooling_schemes=["max"],
    )

    sde_net = SDENet(
        encoder=encoder,
        decoder=decoder,
        drift=drift,
        diffusion=diffusion,
        tasks = None,
        total_time=5.,
        delta_t=1.,
    )

    dummy_data = DummyDataGenerator()

    data = dummy_data.get_data(4)

    decoder = DynEdge(
        nb_inputs=128,
        global_pooling_schemes=["max"],
    )

    print(data)
    print(encoder(data).shape)
    print(sde_net(data))
