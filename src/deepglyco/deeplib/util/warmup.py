# https://github.com/ildoonet/pytorch-gradual-warmup-lr/blob/master/warmup_scheduler/scheduler.py
# Modifications:
# - Update self.last_epoch when self.finished

__all__ = ["GradualWarmupScheduler"]

from typing import Optional, Union

from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler

Scheduler = Union[_LRScheduler, ReduceLROnPlateau]


class GradualWarmupScheduler(_LRScheduler):
    """Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        multiplier: float,
        total_epoch: int,
        after_scheduler: Optional[Scheduler] = None,
    ):
        self.multiplier = multiplier
        if self.multiplier < 1.0:
            raise ValueError("multiplier should be greater thant or equal to 1.")
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.optimizer: Optimizer
        self.base_lrs: list  # _LRScheduler.optimizer and _LRScheduler.base_lrs are not found by type checking
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler and not isinstance(
                self.after_scheduler, ReduceLROnPlateau
            ):
                if not self.finished:
                    self.after_scheduler.base_lrs = [  # type: ignore
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [
                base_lr * (float(self.last_epoch) / self.total_epoch)
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
                for base_lr in self.base_lrs
            ]

    @property
    def requires_metrics(self):
        return isinstance(self.after_scheduler, ReduceLROnPlateau)

    def step(self, epoch=None, metrics=None):
        after_scheduler = self.after_scheduler
        if not isinstance(after_scheduler, ReduceLROnPlateau):
            if self.finished and after_scheduler:
                if epoch is None:
                    self.last_epoch += 1
                    after_scheduler.step(None)
                else:
                    self.last_epoch = epoch
                    after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:

            def step_ReduceLROnPlateau(metrics, epoch=None):
                if epoch is None:
                    epoch = self.last_epoch + 1
                self.last_epoch = (
                    epoch if epoch != 0 else 1
                )  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
                if self.last_epoch <= self.total_epoch:
                    warmup_lr = [
                        base_lr
                        * (
                            (self.multiplier - 1.0) * self.last_epoch / self.total_epoch
                            + 1.0
                        )
                        for base_lr in self.base_lrs
                    ]
                    for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                        param_group["lr"] = lr
                else:
                    if epoch is None:
                        after_scheduler.step(metrics, None)
                    else:
                        after_scheduler.step(metrics, epoch - self.total_epoch)

            step_ReduceLROnPlateau(metrics, epoch)
