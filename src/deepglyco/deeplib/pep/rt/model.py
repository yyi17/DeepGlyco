__all__ = ["PeptideRTBiLSTM", "PeptideRTCNNLSTM"]

import torch
from torch import nn

from ...util.attention import AttentionSum
from ..common.data import PeptideDataBatch
from ..common.model import PeptideBiLSTM, PeptideCNNLSTM


class PeptideRTOutput(nn.Module):
    def __init__(self, configs: dict, in_features: int):
        super().__init__()

        self.attention_sum = AttentionSum(
            nn.Linear(in_features, 1, bias=False)
        )

        self.output = nn.Sequential(
            nn.PReLU(),
            nn.Linear(
                in_features=in_features,
                out_features=1,
            ),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        out_sum = self.attention_sum(feature)
        out = self.output(out_sum)
        out = out.squeeze(1)
        return out


class PeptideRTBiLSTM(PeptideBiLSTM):
    def __init__(self, configs: dict):
        super().__init__(configs)

        self.output = PeptideRTOutput(configs, in_features=super().output_size)

    def forward(self, batch: PeptideDataBatch) -> torch.Tensor:
        x = super().forward(batch)
        y = self.output(x)
        return y


class PeptideRTCNNLSTM(PeptideCNNLSTM):
    def __init__(self, configs: dict):
        super().__init__(configs)

        self.output = PeptideRTOutput(configs, in_features=super().output_size)

    def forward(self, batch: PeptideDataBatch) -> torch.Tensor:
        x = super().forward(batch)
        y = self.output(x)
        return y

