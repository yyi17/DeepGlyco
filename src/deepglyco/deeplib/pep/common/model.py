__all__ = [
    "InputPeptideEncoding",
    "InputPeptideCharge",
    "PeptideBiLSTM",
    "PeptideCNNLSTM",
]

from typing import Optional, cast

import torch
from torch import nn

from ....util.collections.dict import chain_get_typed, chain_item_typed
from .data import PeptideDataBatch


class InputPeptideEncoding(nn.Module):
    def __init__(self, configs: dict):
        super().__init__()

        if chain_get_typed(configs, str, "data", "amino_acids", "encoding") == "onehot":
            self.seq_embedding = None
            self.aa_dim = len(
                chain_item_typed(configs, list, "data", "amino_acids", "symbol")
            )
        else:
            self.aa_dim = chain_item_typed(
                configs, int, "model", "hyperparams", "seq_embedding_dim"
            )
            self.seq_embedding = nn.Embedding(
                len(chain_item_typed(configs, list, "data", "amino_acids", "symbol"))
                + 1,
                embedding_dim=self.aa_dim,
                padding_idx=0,
            )

        if chain_get_typed(configs, dict, "data", "modifications") is None:
            self.mod_embedding = None
            self.mod_dim = 0
        else:
            mod_encoding = chain_get_typed(
                configs, str, "data", "modifications", "encoding"
            )
            if mod_encoding == "onehot":
                self.mod_embedding = None
                self.mod_dim = len(
                    chain_item_typed(configs, list, "data", "modifications", "symbol")
                )
            elif mod_encoding == "atoms":
                self.mod_embedding = None
                self.mod_dim = len(
                    chain_item_typed(configs, list, "data", "modifications", "element")
                )
            else:
                self.mod_dim = chain_item_typed(
                    configs, int, "model", "hyperparams", "mod_embedding_dim"
                )
                self.mod_embedding = nn.Embedding(
                    len(
                        chain_item_typed(
                            configs, list, "data", "modifications", "symbol"
                        )
                    )
                    + 1,
                    embedding_dim=self.mod_dim,
                    padding_idx=0,
                )

    @property
    def output_size(self) -> int:
        return self.aa_dim + self.mod_dim

    def forward(self, batch: PeptideDataBatch) -> torch.Tensor:
        if self.seq_embedding:
            seq = self.seq_embedding(batch.sequence)
        else:
            seq = batch.sequence

        if self.mod_embedding or self.mod_dim:
            if batch.modifications is None:
                raise ValueError("modifications are expected but None given")
            if self.mod_embedding:
                mod = self.mod_embedding(batch.modifications)
            else:
                mod = batch.modifications
            out = torch.cat((seq, mod), 2)
        else:
            out = seq

        return out


class InputPeptideCharge(nn.Module):
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

    def forward(self, batch: PeptideDataBatch) -> Optional[torch.Tensor]:
        if self.charge_dim > 0:
            if batch.charge is None:
                raise ValueError("charge is expected but None given")
            ch = torch.zeros(
                (batch.batch_size, batch.sequence.size(1), self.charge_dim),
                device=batch.charge.device,
            )
            for i, ln in enumerate(batch.length):
                ch[i, :ln, :] = batch.charge[i]
            return ch
        else:
            return None


class PeptideBiLSTM(nn.Module):
    def __init__(self, configs: dict):
        super().__init__()

        self.pep_input = InputPeptideEncoding(configs)
        self.charge_input = InputPeptideCharge(configs)

        lstm_hidden_size = chain_item_typed(
            configs, int, "model", "hyperparams", "lstm_hidden_size"
        )
        lstm_num_layers = chain_item_typed(
            configs, int, "model", "hyperparams", "lstm_num_layers"
        )
        dropout = chain_item_typed(
            configs, float, "model", "hyperparams", "dropout", allow_convert=True
        )

        self.lstm_1 = nn.LSTM(
            input_size=self.pep_input.output_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.lstm_2 = nn.LSTM(
            input_size=lstm_hidden_size * 2 + self.charge_input.charge_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    @property
    def output_size(self) -> int:
        return self.lstm_2.hidden_size * 2

    def forward(self, batch: PeptideDataBatch) -> torch.Tensor:
        pep = self.pep_input(batch)
        pep = pep.float()
        ch = self.charge_input(batch)
        if ch is not None:
            ch = ch.float()

        pep_packed = nn.utils.rnn.pack_padded_sequence(
            pep,
            batch.length,
            batch_first=True,
        )

        pep1_packed, _ = self.lstm_1(pep_packed)
        pep1, _ = nn.utils.rnn.pad_packed_sequence(pep1_packed, batch_first=True)
        pep1 = self.dropout(pep1)

        if ch is not None:
            pep_ch = torch.cat((pep1, ch), dim=-1)
        else:
            pep_ch = pep1

        pep_ch_packed = nn.utils.rnn.pack_padded_sequence(
            pep_ch,
            batch.length,
            batch_first=True,
        )

        out_packed, _ = self.lstm_2(pep_ch_packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        out = self.dropout(out)

        return out


class PeptideCNNLSTM(nn.Module):
    def __init__(self, configs: dict):
        super().__init__()

        self.pep_input = InputPeptideEncoding(configs)
        self.charge_input = InputPeptideCharge(configs)

        conv_num_filters = chain_item_typed(
            configs, int, "model", "hyperparams", "conv_num_filters"
        )
        conv_kernel_size = chain_item_typed(
            configs, int, "model", "hyperparams", "conv_kernel_size"
        )
        if conv_kernel_size is None:
            conv_kernel_size = chain_item_typed(
                configs, list, "model", "hyperparams", "conv_kernel_size"
            )
        lstm_hidden_size = chain_item_typed(
            configs, int, "model", "hyperparams", "lstm_hidden_size"
        )
        lstm_num_layers = chain_item_typed(
            configs, int, "model", "hyperparams", "lstm_num_layers"
        )
        dropout = chain_item_typed(
            configs, float, "model", "hyperparams", "dropout", allow_convert=True
        )

        if not isinstance(conv_kernel_size, list):
            conv_kernel_size = [conv_kernel_size]
        self.conv_list = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=self.pep_input.output_size,
                    out_channels=conv_num_filters,
                    kernel_size=k,
                    padding="same",
                )
                for k in conv_kernel_size
            ]
        )
        self.lstm = nn.LSTM(
            input_size=sum([cast(int, conv.out_channels) for conv in self.conv_list]),
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bidirectional=True,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    @property
    def output_size(self) -> int:
        return self.lstm.hidden_size * 2

    def forward(self, batch: PeptideDataBatch) -> torch.Tensor:
        pep = self.pep_input(batch)
        pep = pep.float()
        ch = self.charge_input(batch)
        if ch is not None:
            ch = ch.float()

        pep_permuted = pep.permute(0, 2, 1)
        pep1_permuted = torch.concat(
            [conv(pep_permuted) for conv in self.conv_list], dim=1
        )
        pep1 = pep1_permuted.permute(0, 2, 1)

        pep1 = self.dropout(pep1)

        if ch is not None:
            pep_ch = torch.cat((pep1, ch), dim=-1)
        else:
            pep_ch = pep1

        pep_ch_packed = nn.utils.rnn.pack_padded_sequence(
            pep_ch,
            batch.length,
            batch_first=True,
        )

        out_packed, _ = self.lstm(pep_ch_packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)
        out = self.dropout(out)
        return out

