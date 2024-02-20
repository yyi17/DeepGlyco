__all__ = ["GlycoPeptideMS2TreeLSTM"]

from typing import Optional
import dgl
import torch
from torch import nn
from dgl.nn.pytorch.glob import GlobalAttentionPooling
from dgl.ops import copy_u_sum  # type: ignore

from ....util.collections.dict import chain_get_typed, chain_item_typed
from ...util.attention import AttentionSum
from ..common.data import GlycoPeptideDataBatch
from ..common.model import GlycoPeptideTreeLSTM
from .data import GlycoPeptideMS2Output


class GPeptideMS2Output(nn.Module):
    def __init__(self, configs: dict, in_features: int):
        super().__init__()

        num_fragment_ion_types = len(
            chain_item_typed(
                configs, list, "data", "peptide_fragments", "fragment_type"
            )
        )
        num_fragment_ion_types *= len(
            chain_item_typed(
                configs, list, "data", "peptide_fragments", "fragment_charge"
            )
        )
        loss_types = chain_get_typed(
            configs, list, "data", "peptide_fragments", "loss_type"
        )
        if loss_types is not None and len(loss_types) > 0:
            raise NotImplementedError(f"loss_type is not supported {loss_types}")

        fragment_glycans = chain_get_typed(
            configs, list, "data", "peptide_fragments", "fragment_glycan"
        )

        def build_output_layer():
            return nn.Sequential(
                nn.Linear(
                    in_features=in_features,
                    out_features=num_fragment_ion_types,
                ),
                torch.nn.ReLU(),
            )

        self.output_main = build_output_layer()

        if fragment_glycans is not None and len(fragment_glycans) > 0:
            self.output_glycan = nn.ModuleDict(
                {gly: build_output_layer() for gly in fragment_glycans}
            )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        out_main = self.output_main(feature)
        out_glycan = [layer(feature) for gly, layer in self.output_glycan.items()]
        out = torch.concat([out_main, *out_glycan], dim=-1)
        return out


class GlycanMS2CleavageNN(nn.Module):
    def __init__(self, in_features: int, out_features: Optional[int] = None):
        super().__init__()
        self.attention_sum = AttentionSum(
            nn.Linear(in_features, 1, bias=False),
            feat_nn=nn.Linear(in_features, out_features, bias=False)
            if out_features is not None
            else None,
        )

    def message_func(self, edges):
        return {"h": edges.src["h"]}

    def reduce_func(self, nodes):
        h_cat = nodes.mailbox["h"]
        h_reduced = self.attention_sum(h_cat)
        return {"h": h_reduced}

    def forward(self, graph: dgl.DGLGraph, feature: torch.Tensor):
        with graph.local_scope():
            graph.srcdata["h"] = feature
            graph.pull(
                graph.dstnodes(),
                message_func=self.message_func,
                reduce_func=self.reduce_func,
            )
            return graph.dstdata.pop("h")


class GlycanMS2FragmentNN(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features, hidden_size=out_features, batch_first=True
        )
        self.attention_sum = AttentionSum(nn.Linear(out_features, 1, bias=False))

    def message_func(self, edges):
        return {"h": edges.src["h"]}

    def reduce_func(self, nodes):
        h_cat = nodes.mailbox["h"]
        if self.training:
            h_cat = h_cat[:, torch.randperm(h_cat.size(1)), :]
        h, _ = self.lstm(h_cat)
        h_reduced = self.attention_sum(h)
        return {"h": h_reduced}

    def forward(self, graph: dgl.DGLGraph, feature: torch.Tensor):
        with graph.local_scope():
            graph.srcdata["h"] = feature
            graph.pull(
                graph.dstnodes(),
                message_func=self.message_func,
                reduce_func=self.reduce_func,
            )
            return graph.dstdata.pop("h")


class GlycanMS2Output(nn.Module):
    def __init__(self, configs: dict, in_features: int):
        super().__init__()

        num_fragment_ion_types = len(
            chain_item_typed(configs, list, "data", "glycan_fragments", "fragment_type")
        )
        num_fragment_ion_types *= len(
            chain_item_typed(
                configs, list, "data", "glycan_fragments", "fragment_charge"
            )
        )

        self.cleavage = nn.ModuleDict(
            {
                k: GlycanMS2CleavageNN(in_features=in_features)
                for k in ["lost", "retained"]
            }
        )
        # self.fragment = GlycanMS2CleavageNN(in_features * 2)
        # self.output = nn.Sequential(
        #     nn.Linear(
        #         in_features=in_features * 2,
        #         out_features=num_fragment_ion_types,
        #     ),
        #     nn.ReLU(),
        # )
        self.fragment = GlycanMS2FragmentNN(in_features * 2, in_features)
        self.output = nn.Sequential(
            nn.Linear(
                in_features=in_features,
                out_features=num_fragment_ion_types,
            ),
            nn.ReLU(),
        )

    def forward(
        self, batch: GlycoPeptideDataBatch, feature: torch.Tensor
    ) -> torch.Tensor:
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
        return gly_out


class GlycoPeptideMS2RatioOutput(nn.Module):
    def __init__(self, configs: dict, pep_features: int, gly_features: int):
        super().__init__()

        self.pep_attention_sum = AttentionSum(nn.Linear(pep_features, 1, bias=False))
        self.gly_attention_sum = GlobalAttentionPooling(
            nn.Linear(gly_features, 1, bias=False)
        )

        self.output = nn.Sequential(
            nn.Linear(
                in_features=pep_features + gly_features,
                out_features=1,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        batch: GlycoPeptideDataBatch,
        pep_feature: torch.Tensor,
        gly_feature: torch.Tensor,
    ) -> torch.Tensor:
        g_link = dgl.edge_type_subgraph(batch.glycan_graph, ["link"])
        g_link.set_batch_num_nodes(batch.glycan_graph.batch_num_nodes("monosaccharide"))
        g_link.set_batch_num_edges(batch.glycan_graph.batch_num_edges("link"))

        pep = self.pep_attention_sum(pep_feature)
        gly = self.gly_attention_sum(g_link, gly_feature)

        pep_gly = torch.concat((pep, gly), dim=-1)
        ratio = self.output(pep_gly).squeeze(1)
        return ratio


class GlycoPeptideMS2TreeLSTM(GlycoPeptideTreeLSTM):
    def __init__(self, configs: dict):
        super().__init__(configs=configs)

        self.pep_output = GPeptideMS2Output(
            configs, in_features=self.pep_lstm_2.hidden_size * 2
        )
        self.gly_output = GlycanMS2Output(
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
        gly_out = self.gly_output(batch, gly)
        ratio = self.ratio_output(batch, pep, gly)
        
        return GlycoPeptideMS2Output(pep_out, gly_out, ratio)
