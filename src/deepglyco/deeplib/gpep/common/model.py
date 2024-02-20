__all__ = ["InputGlycanEncoding", "GlycoPeptideTreeLSTM"]

from typing import Optional
import dgl
import torch
from torch import nn

from ....util.collections.dict import chain_get_typed, chain_item_typed
from ...pep.common.model import InputPeptideCharge, InputPeptideEncoding
from ...util.treelstm import TreeLSTM
from ..common.data import GlycoPeptideDataBatch


class InputGlycanEncoding(nn.Module):
    def __init__(self, configs: dict):
        super().__init__()

        if (
            chain_get_typed(configs, str, "data", "monosaccharides", "encoding")
            == "onehot"
        ):
            self.gly_embedding = None
            self.output_size = len(
                chain_item_typed(configs, list, "data", "monosaccharides", "symbol")
            )
        else:
            self.output_size = chain_item_typed(
                configs, int, "model", "hyperparams", "gly_embedding_dim"
            )
            self.gly_embedding = nn.Embedding(
                len(
                    chain_item_typed(configs, list, "data", "monosaccharides", "symbol")
                )
                + 1,
                embedding_dim=self.output_size,
                padding_idx=0,
            )

    def forward(self, batch: GlycoPeptideDataBatch) -> torch.Tensor:
        if self.gly_embedding:
            gly = self.gly_embedding(batch.monosaccharides)
        else:
            gly = batch.monosaccharides
        return gly


class InputGlycanCharge(nn.Module):
    def __init__(self, configs: dict):
        super().__init__()

        self.charge_dim = chain_get_typed(
            configs,
            int,
            "model",
            "hyperparams",
            "charge_dim",
            default=0,
        )

    def forward(self, batch: GlycoPeptideDataBatch) -> Optional[torch.Tensor]:
        if self.charge_dim > 0:
            if batch.charge is None:
                raise ValueError("charge is expected but None given")
            ch = torch.concat(
                [
                    batch.charge[i].tile(num, self.charge_dim)
                    for i, num in enumerate(
                        batch.glycan_graph.batch_num_nodes("monosaccharide")
                    )
                ],
                dim=0,
            )
            return ch
        else:
            return None


class GlycoPeptideTreeLSTM(nn.Module):
    def __init__(self, configs: dict):
        super().__init__()

        self.pep_input = InputPeptideEncoding(configs)
        self.gly_input = InputGlycanEncoding(configs)
        self.pep_charge_input = InputPeptideCharge(configs)
        self.gly_charge_input = InputGlycanCharge(configs)

        pep_lstm_hidden_size = chain_item_typed(
            configs, int, "model", "hyperparams", "pep_lstm_hidden_size"
        )
        pep_lstm_num_layers = chain_item_typed(
            configs, int, "model", "hyperparams", "pep_lstm_num_layers"
        )
        pep_dropout = chain_item_typed(
            configs, float, "model", "hyperparams", "pep_dropout", allow_convert=True
        )

        gly_lstm_hidden_size = chain_item_typed(
            configs, int, "model", "hyperparams", "gly_lstm_hidden_size"
        )
        gly_lstm_num_layers = chain_item_typed(
            configs, int, "model", "hyperparams", "gly_lstm_num_layers"
        )
        gly_dropout = chain_item_typed(
            configs, float, "model", "hyperparams", "gly_dropout", allow_convert=True
        )

        self.pep_lstm_1 = nn.LSTM(
            input_size=self.pep_input.output_size,
            hidden_size=pep_lstm_hidden_size,
            num_layers=pep_lstm_num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.pep_lstm_2 = nn.LSTM(
            input_size=pep_lstm_hidden_size * 2 + self.pep_charge_input.charge_dim,
            hidden_size=pep_lstm_hidden_size,
            num_layers=pep_lstm_num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.pep_dropout = nn.Dropout(pep_dropout)

        self.gly_lstm_1 = TreeLSTM(
            input_size=self.gly_input.output_size,
            hidden_size=gly_lstm_hidden_size,
            num_layers=gly_lstm_num_layers,
        )
        self.gly_lstm_2 = TreeLSTM(
            input_size=self.gly_input.output_size
            + gly_lstm_hidden_size
            + self.gly_charge_input.charge_dim,
            hidden_size=gly_lstm_hidden_size,
            num_layers=gly_lstm_num_layers,
        )
        self.gly_dropout = nn.Dropout(gly_dropout)

        self.pep_gly_transform = nn.Linear(
            in_features=pep_lstm_hidden_size * 2,
            out_features=gly_lstm_hidden_size,
        )
        self.gly_pep_transform = nn.Linear(
            in_features=gly_lstm_hidden_size,
            out_features=pep_lstm_hidden_size * 2,
        )

    def forward(self, batch: GlycoPeptideDataBatch):
        g_root = torch.concat(
            (
                torch.tensor([0], device=batch.glycan_graph.device),
                batch.glycan_graph.batch_num_nodes("monosaccharide")[:-1].cumsum(dim=0),
            ),
            dim=0,
        )

        pep_0 = self.pep_input(batch)
        pep_0 = pep_0.float()
        pep_ch = self.pep_charge_input(batch)
        if pep_ch is not None:
            pep_ch = pep_ch.float()

        gly_0 = self.gly_input(batch)
        gly_0 = gly_0.float()
        gly_ch = self.gly_charge_input(batch)
        if gly_ch is not None:
            gly_ch = gly_ch.float()

        # peptide 1
        pep_0_packed = nn.utils.rnn.pack_padded_sequence(
            pep_0, batch.length, batch_first=True
        )
        pep_1_packed, (pep_1_hx, _) = self.pep_lstm_1(pep_0_packed)
        pep_1, _ = nn.utils.rnn.pad_packed_sequence(pep_1_packed, batch_first=True)
        pep_1 = self.pep_dropout(pep_1)

        # glycan bottom-up
        g_link_rev = dgl.batch(
            [
                dgl.reverse(gu)
                for gu in dgl.unbatch(
                    dgl.edge_type_subgraph(batch.glycan_graph, ["link"])
                )
            ]
        )
        gly_1 = self.gly_lstm_1(g_link_rev, gly_0)
        gly_1 = self.gly_dropout(gly_1)

        # merge features from peptide 1 to glycan bottom-up, concat charge
        pep_1_hx_glyshape = self.pep_gly_transform(
            pep_1_hx[-2:, :, :].transpose(0, 1).reshape(pep_1_hx.size(1), -1)
        )
        gly_pep = torch.zeros_like(gly_1)
        gly_pep[g_root, :] = pep_1_hx_glyshape
        gly_pep = gly_1 + gly_pep
        gly_pep = torch.concat((gly_0, gly_pep), dim=-1)
        if gly_ch is not None:
            gly_pep_ch = torch.concat((gly_pep, gly_ch), dim=-1)
        else:
            gly_pep_ch = gly_pep

        # glycan top-down
        g_link = dgl.edge_type_subgraph(batch.glycan_graph, ["link"])
        gly_2 = self.gly_lstm_2(g_link, gly_pep_ch)
        gly_2 = self.gly_dropout(gly_2)

        # merge features from glycan bottom-up to peptide 1, concat charge
        gly_1_root_pepshape = self.gly_pep_transform(gly_1[g_root, :])
        pep_gly = torch.zeros_like(pep_1)
        for i in range(g_root.size(0)):
            pep_gly[i, batch.glycan_position[i] - 1, :] = gly_1_root_pepshape[i, :]
        pep_gly = pep_1 + pep_gly
        if pep_ch is not None:
            pep_gly_ch = torch.cat((pep_gly, pep_ch), dim=-1)
        else:
            pep_gly_ch = pep_gly

        # peptide 2
        pep_gly_ch_packed = nn.utils.rnn.pack_padded_sequence(
            pep_gly_ch,
            batch.length,
            batch_first=True,
        )
        pep_2_packed, (_, _) = self.pep_lstm_2(pep_gly_ch_packed)
        pep_2, _ = nn.utils.rnn.pad_packed_sequence(pep_2_packed, batch_first=True)
        pep_2 = self.pep_dropout(pep_2)

        # pep_1_hx_combined = pep_1_hx_glyshape[g_root, :]
        # gly_1_root = gly_1[g_root, :]
        return (
            pep_2,
            gly_2,
        )  # pep_1_hx_combined, gly_1_root
