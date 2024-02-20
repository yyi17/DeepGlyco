__all__ = ["SummaryWriterProto", "TensorBoardSummaryWriter"]

from typing import Protocol

from torch.utils import tensorboard

from ...util.collections.dict import flatten_dict


class SummaryWriterProto(Protocol):
    def write_model_params(self, model, epoch):
        ...

    def write_summary(self, summary, epoch):
        ...

    def write_data_metrics(self, data, epoch):
        ...


class TensorBoardSummaryWriter(tensorboard.writer.SummaryWriter):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def write_model_params(self, model, epoch):
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            self.add_histogram("ModelParams/" + name + "_grad", param.grad, epoch)
            self.add_histogram("ModelParams/" + name + "_data", param, epoch)
        self.flush()

    def write_summary(self, summary, epoch):
        for k, v in flatten_dict(summary, sep="/").items():
            self.add_scalar(k, v, epoch)
        self.flush()

    def write_data_metrics(self, data, epoch):
        for k, v in flatten_dict(data, sep="/").items():
            self.add_histogram(k, v, epoch)
        self.flush()
