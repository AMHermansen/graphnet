"""Callback class(es) for using during model training."""

import os
from typing import Any, Optional

from lightning.pytorch.callbacks import EarlyStopping


class GraphnetEarlyStopping(EarlyStopping):
    """Early stopping callback for graphnet."""

    def __init__(self, save_dir: Optional[str] = None, **kwargs: Any) -> None:
        """Construct `GraphnetEarlyStopping` Callback.

        Args:
            save_dir: Path to directory to save best model and config.
            **kwargs: Keyword arguments to pass to `EarlyStopping`. See
                `pytorch_lightning.callbacks.EarlyStopping` for details.
        """
        self.save_dir = save_dir
        super().__init__(**kwargs)

    def setup(
        self,
        trainer: "pl.Trainer",
        graphnet_model: "Model",
        stage: Optional[str] = None,
    ) -> None:
        """Call at setup stage of training.

        Args:
            trainer: The trainer.
            graphnet_model: The model.
            stage: The stage of training.
        """
        super().setup(trainer, graphnet_model, stage)
        self.save_dir = self.save_dir or trainer.default_root_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", graphnet_model: "Model"
    ) -> None:
        """Call after each train epoch.

        Args:
            trainer: Trainer object.
            graphnet_model: Graphnet Model.

        Returns: None.
        """
        if not self._check_on_train_epoch_end or self._should_skip_check(
            trainer
        ):
            return
        current_best = self.best_score
        self._run_early_stopping_check(trainer)
        if self.best_score != current_best:
            graphnet_model.save_state_dict(
                os.path.join(self.save_dir, "best_model.pth")
            )

    def on_validation_end(
        self, trainer: "pl.Trainer", graphnet_model: "Model"
    ) -> None:
        """Call after each validation epoch.

        Args:
            trainer: Trainer object.
            graphnet_model: Graphnet Model.

        Returns: None.
        """
        if self._check_on_train_epoch_end or self._should_skip_check(trainer):
            return
        current_best = self.best_score
        self._run_early_stopping_check(trainer)
        if self.best_score != current_best:
            graphnet_model.save_state_dict(
                os.path.join(self.save_dir, "best_model.pth")
            )

    def on_fit_end(
        self, trainer: "pl.Trainer", graphnet_model: "Model"
    ) -> None:
        """Call at the end of training.

        Args:
            trainer: Trainer object.
            graphnet_model: Graphnet Model.

        Returns: None.
        """
        graphnet_model.load_state_dict(
            os.path.join(self.save_dir, "best_model.pth")
        )
