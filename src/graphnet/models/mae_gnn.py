from typing import Tuple, List, Union, Optional, Dict, Any

from torch import Tensor
from torch_geometric.data import Data, Batch
from torch_geometric import utils

import torch
from torch.optim import Adam
from torch_geometric.nn import knn_graph

from graphnet.models.gnn.aggregator import Aggregator
from graphnet.models.gnn.gnn import StandardGNN, RawGNN, ValueAwareGNN
from graphnet.models.lightweight_model import LightweightTemplateModel, LightweightModel
from graphnet.models.task import Task


class GMAE(LightweightTemplateModel):
    def __init__(
        self,
        *,
        gnn: RawGNN,
        aggregator: Aggregator,
        tasks: Optional[Union[Task, List[Task]]],
        optimizer_class: type = Adam,
        optimizer_kwargs: Optional[Dict] = None,
        scheduler_class: Optional[type] = None,
        scheduler_kwargs: Optional[Dict] = None,
        scheduler_config: Optional[Dict] = None,
        state_dict_path: Optional[str] = "",
        value_key: str = "track",
        **kwargs,
    ) -> None:
        """Construct lightweight lightning model.

        Args:
            gnn: GNN backbone used.
            tasks: Which task the model is trained for.
            optimizer_class: Optimizer used to train.
            optimizer_kwargs: keyword-args for optimizer.
            scheduler_class: Learning rate scheduler for the optimizer.
            scheduler_kwargs: Scheduler keyword-args.
            scheduler_config: Remaining config for scheduler.
            state_dict_path: Path to state_dict_path to load the model weights.
        """
        # Base class constructor
        super().__init__(name=__name__, class_name=self.__class__.__name__, tasks=tasks, **kwargs)

        # Member variable(s)
        self._gnn = ValueAwareGNN(gnn, aggregator, value_key)
        self._optimizer_class = optimizer_class
        self._optimizer_kwargs = optimizer_kwargs or dict()
        self._scheduler_class = scheduler_class
        self._scheduler_kwargs = scheduler_kwargs or dict()
        self._scheduler_config = scheduler_config or dict()

        if state_dict_path:
            self.load_state_dict(state_dict_path)

    def _apply_masking(
        self, x: torch.Tensor, data: Data
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, L, F = x.shape
        ids_keep, ids_keep_mask = self._get_ids_keep_batch(data)
        ids_restore, ids_restore_mask = self._get_ids_restore_batch(data)
        mask = torch.ones([B, L], device=x.device)
        for i, n in enumerate(data.n_keep):
            mask[i, :n] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return (
            torch.gather(
                x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, F)
            ),
            ids_keep_mask,
            mask.to(bool),  # convert to bool
            ids_restore,
        )

    def _undo_masking(self, data: Data, x: torch.Tensor):
        ids_restore, _ = self._get_ids_restore_batch(data)
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1] + 1, 1
        )
        x_cls = x[:, :1, :]
        x_no_cls = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x = torch.gather(
            x_no_cls,
            dim=1,
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]),
        )  # restore the original order
        x = x + self.decode_pos_embedder(
            torch.arange(x.shape[1], device=x.device)
        )
        x = torch.cat([x_cls, x_no_cls], dim=1)  # prepend cls token
        return x

    def _get_ids_keep_batch(
        self, data: Data
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ids_keep_batch = self.custom_repeat(data.n_keep)
        return utils.to_dense_batch(data.ids_keep, ids_keep_batch)

    def _get_ids_restore_batch(
        self, data: Data
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return utils.to_dense_batch(data.ids_restore, data.batch)

    def _shared_step(self, data: Data):
        x_orig, padding_mask = utils.to_dense_batch(data.x, data.batch)
        x, ids_keep_mask, mask, ids_restore = self._apply_masking(x_orig, data)

        new_data_list = [Data(x[batch_number, :][ids_keep_mask[batch_number]]) for batch_number in range(x.shape[0])]
        new_data = Batch.from_data_list(new_data_list)
        new_data.edge_index = knn_graph(
            new_data.x,
            8,
            new_data.batch,
        )

        for key, value in data.items():
            if key in new_data:
                new_data[f"old_{key}"] = value
            else:
                new_data[key] = value
        new_data.ae_mask = mask
        preds = self._gnn(new_data)
        preds = [task(preds) for task in self._tasks]

        return preds, new_data

    def _compute_loss(
        self, preds: List[Tensor], new_data: Data, verbose: bool = False
    ) -> Tensor:
        # Testing
        losses = [
            task.compute_loss(pred, new_data)
            for task, pred in zip(self._tasks, preds)
        ]
        return torch.mean(torch.stack(losses))

    def training_step(self, train_batch: Data, batch_idx: int):
        preds, new_data = self._shared_step(train_batch)
        # self._update_train_metrics(preds, new_data)
        loss = self._compute_loss(preds, new_data)
        self.log(
            "train_loss",
            loss,
            batch_size=self._get_batch_size(train_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        # self._log_metrics(
        #      *self.train_metrics,
        #      batch_size=self._get_batch_size(train_batch),
        #      prog_bar=True,
        #      on_epoch=True,
        #      on_step=False,
        #      sync_dist=True,
        # )
        return {"loss": loss, "preds": preds, "new_data": new_data}

    def validation_step(
        self, val_batch: Data, batch_idx: int
    ) -> Dict[str, Any]:
        """Perform validation step."""
        preds, new_data = self._shared_step(val_batch)
        self._update_val_metrics(preds, new_data)
        loss = self._compute_loss(preds, new_data)
        self.log(
            "val_loss",
            loss,
            batch_size=self._get_batch_size(val_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        self._log_metrics(
            *self.val_metrics,
            batch_size=self._get_batch_size(val_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
        return {"loss": loss, "preds": preds, "new_data": new_data}

    def test_step(self, test_batch: Data, batch_idx: int) -> Dict[str, Any]:
        """Perform test step."""
        preds, new_data = self._shared_step(test_batch)
        loss = self._compute_loss(preds, new_data)
        self.log(
            "test_loss",
            loss,
            batch_size=self._get_batch_size(test_batch),
            prog_bar=True,
            on_epoch=True,
            on_step=False,
        )
        return {"loss": loss, "preds": preds, "new_data": new_data}

    def predict_step(self, predict_batch: Data, batch_idx: int) -> Dict[str, Any]:
        preds, new_data = self._shared_step(predict_batch)
        return {
            "preds": preds,
            "new_data": new_data
        }

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure the model's optimizer(s)."""
        optimizer = self._optimizer_class(
            self.parameters(), **self._optimizer_kwargs
        )
        config = {
            "optimizer": optimizer,
        }
        if self._scheduler_class is not None:
            scheduler = self._scheduler_class(
                optimizer, **self._scheduler_kwargs
            )
            config.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        **self._scheduler_config,
                    },
                }
            )
        return config

    @staticmethod
    def custom_repeat(tensor: torch.Tensor) -> torch.Tensor:
        """Create batch indices from the number of nodes in each graph.

        Args:
            tensor: Number of nodes in each graph.

        Returns:
            Batch indices.
        """
        result: List[torch.Tensor] = []
        for i, n in enumerate(tensor):
            repeated_sequence = torch.full(
                (n,), i, dtype=torch.int64, device=tensor.device
            )
            result.append(repeated_sequence)
        return torch.cat(result)


class FinetuneMAE(LightweightModel):
    def __init__(
        self,
        *,
        gnn: RawGNN,
        aggregator: Aggregator, 
        mae_ckpt: Optional[str] = None,
        tasks: Optional[Union[Task, List[Task]]],
        value_key: str = "track",
        **kwargs,
    ) -> None:
        """Construct lightweight lightning model.

        Args:
            gnn: GNN backbone used.
            tasks: Which task the model is trained for.
            optimizer_class: Optimizer used to train.
            optimizer_kwargs: keyword-args for optimizer.
            scheduler_class: Learning rate scheduler for the optimizer.
            scheduler_kwargs: Scheduler keyword-args.
            scheduler_config: Remaining config for scheduler.
            state_dict_path: Path to state_dict_path to load the model weights.
        """
        # Base class constructor
        super().__init__(
            tasks=tasks,
            gnn=gnn,
            aggregator=aggregator,
            **kwargs
        )

        # Member variable(s)
        self._gnn = ValueAwareGNN(gnn, aggregator, value_key)

        if mae_ckpt is not None: 
            state_dict = torch.load(mae_ckpt)["state_dict"]
            state_dict = {key: value for key, value in state_dict.items() if "task" not in key}
            self.load_state_dict(state_dict, strict=False)
        


# Dirty testing
if __name__ == "__main__":
    from graphnet.models.gnn import DynEdgeTITO
    from graphnet.models.gnn.aggregator import StandardPooling
    from graphnet.data.datamodule import ShardedDataModule, ShardedDatasetConfig
    from graphnet.models.graphs import GraphDefinition
    from graphnet.models.graphs.nodes.nodes import MAENodes
    from graphnet.models.graphs.edges.edges import NoEdges, KNNEdges
    from graphnet.models.detector import IceCube86
    from graphnet.models.task import MAEClassificationTask, RawZenithReconstructionWithKappa
    from graphnet.training.loss_functions import CosZenithAdjustedVMF

    mae = GMAE(
        gnn=(de := DynEdgeTITO(7)),
        aggregator=(pool := StandardPooling(7, de.nb_outputs, global_pooling_schemes=["mean"])),
        tasks=MAEClassificationTask(pool.nb_outputs)
    )

    data = ShardedDataModule(
        GraphDefinition(IceCube86(), MAENodes(128, 0.5), KNNEdges(8), add_sensor_id=True),
        ShardedDatasetConfig(
            "/lustre/hpc/icecube/andreash/workspace/data_scripts/parquet_sharded/truth.parquet",
            "SplitInIcePulses",
            truth=None,
            features=["dom_x", "dom_y", "dom_z", "dom_time", "charge", "rde", "pmt_area"],
        ),
        selections=dict(
            train="/lustre/hpc/icecube/andreash/workspace/data_scripts/parquet_sharded/selections/cascade_train.parquet",
            val="/lustre/hpc/icecube/andreash/workspace/data_scripts/parquet_sharded/selections/cascade_val.parquet",
        ),
        common_loader_config={
            "batch_size": 256,
            "num_workers": 20,
        }
    )

    dl = data.train_dataloader()
    data_input = next(iter(dl))
    from icecream import ic
    ic(data_input)

    ft = FinetuneMAE(
        gnn=(de := DynEdgeTITO(7)),
        aggregator=(pool := StandardPooling(7, de.nb_outputs, global_pooling_schemes=["mean"])),
        tasks=RawZenithReconstructionWithKappa(hidden_size=pool.nb_outputs, loss_function=CosZenithAdjustedVMF()),
        mae_ckpt="/groups/icecube/andreash/workspace/training/outputs/ood_impact/mae/wandb_continued/mae_pretrain/bwddg19m/checkpoints/epoch=7-step=18752.ckpt"
    )
    ic(ft.training_step(data_input, 0))
