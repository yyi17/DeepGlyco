__all__ = ["TrainerBase"]

import abc
import os
from logging import Logger
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Mapping,
    Optional,
    Sequence,
    Sized,
    TypeVar,
    Union,
)

import torch
from torch.optim.lr_scheduler import (
    _LRScheduler,
    SequentialLR,
    LinearLR,
    ReduceLROnPlateau,
)
from torch.utils.data import DataLoader, Dataset, random_split

from deepglyco.deeplib.util.warmup import GradualWarmupScheduler

from ...util.config import Configurable
from ...util.collections.dict import chain_get, chain_get_typed, chain_item_typed
from ...util.progress import ProgressFactoryProto, ProgressProto
from ..common.data import BatchProto
from ..util.earlystop import EarlyStopping
from ..util.writer import SummaryWriterProto

DataConverterType = TypeVar("DataConverterType")
DataType = TypeVar("DataType")


class TrainerBase(abc.ABC, Generic[DataType], Configurable):
    def __init__(
        self,
        configs: Union[str, dict],
        progress_factory: Optional[ProgressFactoryProto] = None,
        summary_writer: Optional[SummaryWriterProto] = None,
        logger: Optional[Logger] = None,
    ):
        super().__init__(configs)
        self.progress_factory = progress_factory
        self.summary_writer = summary_writer
        self.logger = logger
        self.last_epoch = -1

    def load_model(self, model: Union[str, dict]):
        if isinstance(model, str):
            pretrained: dict = torch.load(model)
        else:
            pretrained = model
        self.set_configs(chain_item_typed(pretrained, dict, "config"))

        model_state = chain_item_typed(pretrained, dict, "state")
        self.build_model()
        self.model.load_state_dict(model_state)

        loss_state = chain_get_typed(pretrained, dict, "loss_state")
        if loss_state is not None:
            self.loss_fn.load_state_dict(loss_state)

    @abc.abstractmethod
    def _collate_data(self, batch_data: List[DataType]) -> BatchProto[DataType]:
        pass

    def load_data(
        self,
        dataset: Dataset[DataType],
        val_ratio: float = 1.0 / 3,
        holdout_ratio: float = 0.0,
        seed: Optional[int] = None,
    ):
        assert isinstance(dataset, Sized)
        assert val_ratio >= 0 and holdout_ratio >= 0 and val_ratio + holdout_ratio <= 1

        generator = torch.Generator().manual_seed(seed) if seed is not None else None

        dataset_size = len(dataset)
        val_size = int(val_ratio * dataset_size)
        train_size = dataset_size - val_size

        if holdout_ratio > 0:
            holdout_size = int(holdout_ratio * dataset_size)
            train_size -= holdout_size
            train_dataset, val_dataset, holdout_dataset = random_split(
                dataset, [train_size, val_size, holdout_size], generator=generator
            )
        else:
            train_dataset, val_dataset = random_split(
                dataset, [train_size, val_size], generator=generator
            )

        batch_size = self.get_config("training", "batch_size", typed=int)
        self.train_loader = DataLoader[DataType](
            train_dataset, batch_size=batch_size, collate_fn=self._collate_data
        )
        self.val_loader = DataLoader[DataType](
            val_dataset, batch_size=batch_size, collate_fn=self._collate_data
        )
        return train_dataset.indices, val_dataset.indices

    @abc.abstractmethod
    def _build_model(self) -> torch.nn.Module:
        pass

    @abc.abstractmethod
    def _build_loss_fn(self) -> torch.nn.Module:
        pass

    def _build_metrics_fn(self) -> Mapping[str, torch.nn.Module]:
        return {}

    @property
    def default_metrics(self) -> Optional[str]:
        return None

    def _build_optimizer(self) -> torch.optim.Optimizer:
        lr = self.get_config("training", "lr", typed=float, allow_convert=True)
        params = self.model.parameters()

        if any(map(lambda _: True, self.loss_fn.parameters())):
            params = [
                {"params": params},
                {"params": self.loss_fn.parameters(), "weight_decay": 0},
            ]

        return torch.optim.Adam(params, lr=lr, weight_decay=0.0001)

    def _build_scheduler(self) -> Union[_LRScheduler, ReduceLROnPlateau, None]:
        scheduler = None
        if (
            self.get_config("training", "scheduler", required=False, typed=dict)
            is not None
        ):
            scheduler_type = self.get_config("training", "scheduler", "type", typed=str)
            scheduler_factory = getattr(torch.optim.lr_scheduler, scheduler_type, None)
            if scheduler_factory is None or not issubclass(
                scheduler_factory, (_LRScheduler, ReduceLROnPlateau)
            ):
                raise TypeError(f"invalid scheduler type {scheduler_type}")
            scheduler = scheduler_factory(
                self.optimizer,
                **(
                    self.get_config(
                        "training", "scheduler", "args", required=False, typed=dict
                    )
                    or {}
                ),
            )

        warmup = self.get_config("training", "warmup", required=False, typed=int) or 0
        # if warmup > 0:
        #     if scheduler is None:
        #         scheduler = LinearLR(
        #             self.optimizer,
        #             start_factor=0,
        #             end_factor=1,
        #             total_iters=warmup,
        #         )
        #     elif isinstance(scheduler, _LRScheduler):
        #         scheduler = SequentialLR(
        #             self.optimizer,
        #             schedulers=[
        #                 LinearLR(
        #                     self.optimizer,
        #                     start_factor=0,
        #                     end_factor=1,
        #                     total_iters=warmup,
        #                 ),
        #                 scheduler,
        #             ],
        #             milestones=[warmup],
        #         )
        #     else:
        #         raise TypeError(f"warmup with {type(scheduler)} not supported")

        if warmup > 0:
            scheduler = GradualWarmupScheduler(
                self.optimizer,
                multiplier=1,
                total_epoch=warmup,
                after_scheduler=scheduler,
            )
        return scheduler

    def build_model(self):
        device = self.get_config("device", required=False, typed=str)
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)
        self.device = device

        self.model = self._build_model().to(self.device)
        self.loss_fn = self._build_loss_fn()
        self.metrics_fn = self._build_metrics_fn()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

    def save_model(self, file: str):
        data = {
            "state": self.model.state_dict(),
            "config": {
                k: v for k, v in self.get_configs().items() if k in {"data", "model"}
            },
        }

        loss_state = self.loss_fn.state_dict()
        if len(loss_state) > 0:
            data["loss_state"] = loss_state

        torch.save(data, file)

    def _calculate_loss_metrics(
        self,
        batch: BatchProto[DataType],
        pred,
        keep_all_loss_metrics: bool = False,
    ) -> dict:
        loss = self.loss_fn(batch, pred)

        r = {}
        if len(loss.shape) > 0:
            if keep_all_loss_metrics:
                loss_all = torch.empty_like(loss)
                for i in range(loss.size(0)):
                    loss_all[batch.indices[i]] = loss[i]
                r["loss_all"] = loss_all.detach().cpu()
            loss = loss.nanmean()
        r["loss"] = loss

        if self.metrics_fn:
            metrics = {}
            metrics_all = {}
            for k, v in self.metrics_fn.items():
                m = v(batch, pred)
                if len(m.shape) > 0:
                    if keep_all_loss_metrics:
                        m_all = torch.empty_like(m)
                        for i in range(m.size(0)):
                            m_all[batch.indices[i]] = m[i]
                        metrics_all[k] = m_all.detach().cpu()
                    m = m.nanmean()
                metrics[k] = m
            if metrics:
                r["metrics"] = metrics
            if metrics_all:
                r["metrics_all"] = metrics_all
        return r

    def train_one_epoch(self, keep_all_loss_metrics: bool = False) -> dict:
        if self.progress_factory:
            data = self.progress_factory(
                self.train_loader,
                total=len(self.train_loader),
                desc=f"Epoch {self.last_epoch} training",
            )
        else:
            data = self.train_loader

        self.model.train()
        running_loss = 0.0
        loss_all = []
        running_metrics = {}
        metrics_all = []
        current = 0
        for batch in data:
            batch: BatchProto[DataType] = batch.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(batch)
            loss_metrics = self._calculate_loss_metrics(
                batch, pred, keep_all_loss_metrics=keep_all_loss_metrics
            )

            loss: torch.Tensor = loss_metrics["loss"]
            # assert not torch.isnan(loss).any().item()
            loss.backward()

            # assert (
            #     not torch.stack([torch.isnan(p).any() for p in self.model.parameters()])
            #     .any()
            #     .item()
            # )
            self.optimizer.step()
            # assert (
            #     not torch.stack([torch.isnan(p).any() for p in self.model.parameters()])
            #     .any()
            #     .item()
            # )

            current += batch.batch_size
            running_loss += (loss.item() - running_loss) * batch.batch_size / current

            loss_all_batch = loss_metrics.get("loss_all", None)
            if loss_all_batch is not None:
                loss_all.append(loss_all_batch)

            metrics: dict[str, torch.Tensor] = loss_metrics.get("metrics", None)
            if metrics is not None:
                for k, v in metrics.items():
                    s = running_metrics.get(k, 0.0)
                    running_metrics[k] = s + (
                        (v.item() - s) * batch.batch_size / current
                    )
            metrics_all_batch = loss_metrics.get("metrics_all", None)
            if metrics_all_batch is not None:
                metrics_all.append(metrics_all_batch)

            if isinstance(data, ProgressProto):
                data.set_postfix(loss=running_loss, **running_metrics)

        r = {"loss": running_loss}
        if loss_all:
            r["loss_all"] = torch.concat(loss_all)
        if running_metrics:
            r["metrics"] = running_metrics
        if metrics_all:
            r["metrics_all"] = {
                k: torch.concat([x[k] for x in metrics_all]) for k in metrics_all[0]
            }
        return r

    def validate(self, keep_all_loss_metrics: bool = True) -> dict:
        if self.progress_factory:
            data = self.progress_factory(
                self.val_loader,
                total=len(self.val_loader),
                desc=f"Epoch {self.last_epoch} validating",
            )
        else:
            data = self.val_loader

        self.model.eval()
        running_loss = 0.0
        loss_all = []
        running_metrics = {}
        metrics_all = []
        current = 0
        with torch.no_grad():
            for batch in data:
                batch: BatchProto[DataType] = batch.to(self.device)

                pred = self.model(batch)
                loss_metrics = self._calculate_loss_metrics(
                    batch, pred, keep_all_loss_metrics=keep_all_loss_metrics
                )

                loss: torch.Tensor = loss_metrics["loss"]
                current += batch.batch_size
                running_loss += (
                    (loss.item() - running_loss) * batch.batch_size / current
                )

                loss_all_batch = loss_metrics.get("loss_all", None)
                if loss_all_batch is not None:
                    loss_all.append(loss_all_batch)

                metrics: dict[str, torch.Tensor] = loss_metrics.get("metrics", None)
                if metrics is not None:
                    for k, v in metrics.items():
                        s = running_metrics.get(k, 0.0)
                        running_metrics[k] = s + (
                            (v.item() - s) * batch.batch_size / current
                        )
                metrics_all_batch = loss_metrics.get("metrics_all", None)
                if metrics_all_batch is not None:
                    metrics_all.append(metrics_all_batch)

                if isinstance(data, ProgressProto):
                    data.set_postfix(loss=running_loss, **running_metrics)

        r = {"loss": running_loss}
        if loss_all:
            r["loss_all"] = torch.concat(loss_all)
        if running_metrics:
            r["metrics"] = running_metrics
        if metrics_all:
            r["metrics_all"] = {
                k: torch.concat([x[k] for x in metrics_all]) for k in metrics_all[0]
            }
        return r

    def train(
        self,
        num_epochs: int,
        patience: int = 0,
        checkpoint: Optional[str] = "checkpoints/epoch_{epoch}.pt",
        summary_epoch_interval: int = 5,
        callback: Optional[Callable[[Dict], Any]] = None,
    ) -> dict:
        earlystop = EarlyStopping(patience=patience) if patience else None

        best = {
            "epoch": -1,
            "loss": {"training": torch.inf, "validation": torch.inf},
        }
        for epoch in range(num_epochs):
            self.last_epoch += 1
            if self.summary_writer is None:
                keep_all_loss_metrics = False
            elif self.last_epoch % summary_epoch_interval == 0:
                keep_all_loss_metrics = True
            else:
                keep_all_loss_metrics = False

            train_result = self.train_one_epoch(
                keep_all_loss_metrics=keep_all_loss_metrics
            )
            val_result = self.validate(keep_all_loss_metrics=keep_all_loss_metrics)
            summary = {
                k: {"training": train_result[k], "validation": val_result[k]}
                for k in train_result
                if not k.endswith("_all") and k in val_result
            }
            if self.scheduler:
                summary["hyperparams"] = {
                    "learning_rate": self.optimizer.param_groups[0]["lr"]
                }

            if self.logger:
                self.logger.info(
                    f"Epoch {self.last_epoch}: training loss {train_result['loss']}, validation loss {val_result['loss']}"
                )

                if "metrics" in train_result and "metrics" in val_result:
                    self.logger.info(
                        f"Epoch {self.last_epoch}: training metrics "
                        + " ".join(
                            [f"{k}={v}" for k, v in train_result["metrics"].items()]
                        )
                        + ", validation metrics "
                        + " ".join(
                            [f"{k}={v}" for k, v in val_result["metrics"].items()]
                        )
                    )

            if self.default_metrics is None:
                default_metrics = val_result["loss"]
                default_metrics_best = chain_get(best, "loss", "validation", default=torch.inf)
            else:
                default_metrics = val_result["metrics"][self.default_metrics]
                default_metrics_best = chain_get(best, "metrics", "validation",self.default_metrics, default=torch.inf)
            if default_metrics < default_metrics_best:
                checkpoint_path = (
                    checkpoint.format(epoch=self.last_epoch) if checkpoint else None
                )

                if self.logger:
                    self.logger.info(
                        f"Epoch {self.last_epoch}: validation {'loss' if self.default_metrics is None else self.default_metrics} "
                        f"{default_metrics} < best {default_metrics_best}"
                        + (
                            f", saving checkpoint {checkpoint_path}"
                            if checkpoint_path
                            else ""
                        )
                    )

                if checkpoint_path:
                    if not os.path.exists(os.path.dirname(checkpoint_path)):
                        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                    self.save_model(checkpoint_path)
                    best["checkpoint"] = checkpoint_path
                best["epoch"] = self.last_epoch
                best.update(summary)

            if self.summary_writer:
                self.summary_writer.write_summary(summary, self.last_epoch)
                if keep_all_loss_metrics:
                    data_metrics = {
                        k: {
                            "training": train_result[k],
                            "validation": val_result[k],
                        }
                        for k in train_result
                        if k.endswith("_all") and k in val_result
                    }
                    self.summary_writer.write_data_metrics(
                        data_metrics, self.last_epoch
                    )
                    self.summary_writer.write_model_params(self.model, self.last_epoch)

            if callback:
                data_metrics = {
                    k: {
                        "training": train_result[k],
                        "validation": val_result[k],
                    }
                    for k in train_result
                    if k.endswith("_all") and k in val_result
                }
                summary.update(data_metrics)
                callback(summary)

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(metrics=default_metrics)
            elif self.scheduler:
                self.scheduler.step()

            if earlystop and earlystop.step(default_metrics):
                if self.logger:
                    self.logger.info("Early stopped")
                break

        return best
