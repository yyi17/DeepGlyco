__all__ = ["GlycoPeptideBranchMS2TreeLSTM"]

import dgl
import torch
from torch import nn
from dgl.ops import copy_u_sum # type: ignore

from ..ms2.model import GPeptideMS2Output, GlycanMS2FragmentNN, GlycanMS2Output, GlycoPeptideMS2RatioOutput
from ....util.collections.dict import chain_item_typed
from ..common.data import GlycoPeptideDataBatch
from ..common.model import GlycoPeptideTreeLSTM
from .data import GlycoPeptideBranchMS2Output


class GlycanBranchMS2Output(GlycanMS2Output):
    def __init__(self, configs: dict, in_features: int):
        super().__init__(configs, in_features=in_features)

        num_fragment_ion_types = len(
            chain_item_typed(configs, list, "data", "glycan_branch_fragments", "fragment_type")
        )
        num_fragment_ion_types *= len(
            chain_item_typed(
                configs, list, "data", "glycan_branch_fragments", "fragment_charge"
            )
        )

        self.fragment_branch = GlycanMS2FragmentNN(in_features * 2, in_features)
        self.output_branch = nn.Sequential(
            nn.Linear(
                in_features=in_features,
                out_features=num_fragment_ion_types,
            ),
            nn.ReLU(),
        )

    def forward(
        self, batch: GlycoPeptideDataBatch, feature: torch.Tensor
    ):
        gly_cleavage = torch.concat(
            [
                module(dgl.edge_type_subgraph(batch.glycan_graph, [edgetype]), feature)
                for edgetype, module in self.cleavage.items()
            ],
            dim=-1,
        )
        gly_fragment = self.fragment(
            dgl.edge_type_subgraph(batch.glycan_graph, ["join"]), gly_cleavage
        )
        gly_fragment_out = self.output(gly_fragment)
        gly_out = copy_u_sum(
            dgl.edge_type_subgraph(batch.glycan_graph, ["combine"]), gly_fragment_out
        )
        glyb_fragment = self.fragment_branch(
            dgl.edge_type_subgraph(batch.glycan_graph, ["branch_join"]), gly_cleavage
        )
        glyb_fragment_out = self.output_branch(glyb_fragment)
        glyb_out = copy_u_sum(
            dgl.edge_type_subgraph(batch.glycan_graph, ["branch_combine"]), glyb_fragment_out
        )
        return gly_out, glyb_out


class GlycoPeptideBranchMS2TreeLSTM(GlycoPeptideTreeLSTM):
    def __init__(self, configs: dict):
        super().__init__(configs=configs)

        self.pep_output = GPeptideMS2Output(
            configs, in_features=self.pep_lstm_2.hidden_size * 2
        )
        self.gly_output = GlycanBranchMS2Output(
            configs, in_features=self.gly_lstm_2.hidden_size
        )

        self.ratio_output = GlycoPeptideMS2RatioOutput(
            configs,
            pep_features=self.pep_lstm_1.hidden_size * 2,
            gly_features=self.gly_lstm_1.hidden_size,
        )

    def forward(self, batch: GlycoPeptideDataBatch):
        pep, gly = super().forward(batch)

        pep_out = self.pep_output(pep)
        gly_out, glyb_out = self.gly_output(batch, gly)
        ratio = self.ratio_output(batch, pep, gly)

        return GlycoPeptideBranchMS2Output(pep_out, gly_out, glyb_out, ratio)
