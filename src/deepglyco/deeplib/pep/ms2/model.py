__all__ = ["PeptideMS2Output", "PeptideMS2BiLSTM", "PeptideMS2CNNLSTM"]

import torch
from torch import nn

from ....util.collections.dict import chain_get_typed, chain_item_typed
from ...util.timedistributed import TimeDistributed
from ..common.data import PeptideDataBatch
from ..common.model import PeptideBiLSTM, PeptideCNNLSTM


class PeptideMS2Output(nn.Module):
    def __init__(self, configs: dict, in_features: int):
        super().__init__()

        num_fragment_ion_types = len(
            chain_item_typed(configs, list, "data", "fragments", "fragment_type")
        )
        num_fragment_ion_types *= len(
            chain_item_typed(configs, list, "data", "fragments", "fragment_charge")
        )
        loss_types = chain_get_typed(configs, list, "data", "fragments", "loss_type")

        def build_output_layer():
            return nn.Sequential(
                nn.Linear(
                    in_features=in_features,
                    out_features=num_fragment_ion_types,
                ),
                torch.nn.ReLU(),
            )

        self.output_main = build_output_layer()

        if loss_types is not None and len(loss_types) > 0:
            self.output_loss = nn.ModuleDict(
                {loss: build_output_layer() for loss in loss_types}
            )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        out_main = self.output_main(feature)
        out_loss = [layer(feature) for loss, layer in self.output_loss.items()]
        out = torch.concat([out_main, *out_loss], dim=-1)
        return out


class PeptideMS2BiLSTM(PeptideBiLSTM):
    def __init__(self, configs: dict):
        super().__init__(configs)

        self.output = PeptideMS2Output(configs, super().output_size)

    def forward(self, batch: PeptideDataBatch) -> torch.Tensor:
        x = super().forward(batch)
        y = self.output(x)
        return y


class PeptideMS2CNNLSTM(PeptideCNNLSTM):
    def __init__(self, configs: dict):
        super().__init__(configs)
        self.output = PeptideMS2Output(configs, super().output_size)

    def forward(self, batch: PeptideDataBatch) -> torch.Tensor:
        x = super().forward(batch)
        y = self.output(x)
        return y

