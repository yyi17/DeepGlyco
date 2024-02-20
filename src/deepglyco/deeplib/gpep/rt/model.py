__all__ = ["GlycoPeptideRTTreeLSTM"]


from typing import Any, Mapping
import torch
from dgl.nn.pytorch.glob import GlobalAttentionPooling
from torch import nn

from ...util.attention import AttentionSum
from ..common.data import GlycoPeptideDataBatch
from ..common.model import GlycoPeptideTreeLSTM


class GlycoPeptideRTOutput(nn.Module):
    def __init__(self, configs: dict, pep_features: int, gly_features: int):
        super().__init__()

        self.pep_attention_sum = AttentionSum(nn.Linear(pep_features, 1, bias=False))
        self.gly_attention_sum = GlobalAttentionPooling(
            nn.Linear(gly_features, 1, bias=False)
        )
        self.pep_output = nn.Sequential(
            nn.PReLU(),
            nn.Linear(
                in_features=pep_features,
                out_features=1,
            ),
        )
        self.gly_output = nn.Sequential(
            nn.PReLU(),
            nn.Linear(
                in_features=gly_features,
                out_features=1,
            ),
        )

    def forward(
        self,
        batch: GlycoPeptideDataBatch,
        pep_feature: torch.Tensor,
        gly_feature: torch.Tensor,
    ) -> torch.Tensor:
        pep = self.pep_attention_sum(pep_feature)
        gly = self.gly_attention_sum(batch.glycan_graph, gly_feature)
        pep_out = self.pep_output(pep)
        gly_out = self.gly_output(gly)
        out = pep_out + gly_out
        out = out.squeeze(1)
        return out


class GlycoPeptideRTTreeLSTM(GlycoPeptideTreeLSTM):
    def __init__(self, configs: dict):
        super().__init__(configs)

        self.output = GlycoPeptideRTOutput(
            configs=configs,
            pep_features=self.pep_lstm_2.hidden_size * 2,
            gly_features=self.gly_lstm_2.hidden_size,
        )

    def forward(self, batch: GlycoPeptideDataBatch):
        pep, gly = super().forward(batch)

        out = self.output(batch, pep, gly)
        return out

