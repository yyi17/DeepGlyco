__all__ = [
    "GlycoPeptideRTData",
    "GlycoPeptideRTDataBatch",
    "GlycoPeptideRTDataConverter",
    "GlycoPeptideRTDataset",
]

from typing import Any, List, NamedTuple, Optional, Protocol, Union, cast

import dgl
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence

from ....chem.gpep.glycans import GlycanNode, MonosaccharideCollection
from ....chem.pep.mods import ModifiedSequenceParser
from ...pep.rt.data import PeptideRTDataConverter, PeptideRTOutputConverter, RTDataset
from ..common.data import GlycoPeptideDataConverter


class GlycoPeptideRTData(NamedTuple):
    sequence: torch.Tensor
    length: torch.Tensor
    modifications: Optional[torch.Tensor]
    glycan_graph: dgl.DGLGraph
    glycan_position: torch.Tensor
    retention_time: torch.Tensor

    @property
    def monosaccharides(self):
        return self.glycan_graph.nodes["monosaccharide"].data["monosaccharide"]

    def to(self, device):
        return self.__class__(
            sequence=self.sequence.to(device),
            length=self.length,
            modifications=self.modifications.to(device)
            if self.modifications is not None
            else None,
            glycan_graph=self.glycan_graph.to(device),
            glycan_position=self.glycan_position.to(device),
            retention_time=self.retention_time.to(device),
        )


class GlycoPeptideRTDataBatch(NamedTuple):
    sequence: torch.Tensor
    length: torch.Tensor
    modifications: Optional[torch.Tensor]
    glycan_graph: dgl.DGLGraph
    glycan_position: torch.Tensor
    retention_time: torch.Tensor
    indices: torch.Tensor

    @property
    def monosaccharides(self):
        return self.glycan_graph.nodes["monosaccharide"].data["monosaccharide"]

    @property
    def batch_size(self):
        return self.indices.size(0)

    def to(self, device):
        return self.__class__(
            sequence=self.sequence.to(device),
            length=self.length,
            modifications=self.modifications.to(device)
            if self.modifications is not None
            else None,
            glycan_graph=self.glycan_graph.to(device),
            glycan_position=self.glycan_position.to(device),
            retention_time=self.retention_time.to(device),
            indices=self.indices,
        )

    @classmethod
    def collate(cls, batch_data: List[GlycoPeptideRTData]):
        length = torch.as_tensor([x.length for x in batch_data])
        indices = torch.argsort(length, descending=True)
        batch_data = [batch_data[i] for i in indices]
        length = length[indices]

        graph = dgl.batch([x.glycan_graph for x in batch_data])

        modifications = [
            x.modifications for x in batch_data if x.modifications is not None
        ]
        if len(modifications) == 0:
            modifications = None
        elif len(modifications) != len(batch_data):
            raise ValueError(f"modifications contains None")

        return cls(
            sequence=pad_sequence([x.sequence for x in batch_data], batch_first=True),
            length=length,
            modifications=pad_sequence(modifications, batch_first=True)
            if modifications is not None
            else None,
            glycan_graph=graph,
            glycan_position=torch.as_tensor([x.glycan_position for x in batch_data]),
            retention_time=torch.as_tensor([x.retention_time for x in batch_data]),
            indices=indices,
        )


class GlycoPeptideRTEntryProto(Protocol):
    modified_sequence: str
    glycan_struct: str
    glycan_position: int
    retention_time: float


class GlycoPeptideRTDataConverter(GlycoPeptideDataConverter):
    def __init__(
        self,
        sequence_parser: ModifiedSequenceParser,
        monosaccharides: MonosaccharideCollection,
        configs: Union[str, dict],
    ):
        super().__init__(sequence_parser, monosaccharides, configs)

    def rt_entry_to_tensor(self, entry: GlycoPeptideRTEntryProto) -> GlycoPeptideRTData:
        parsed_sequence = self.sequence_parser.parse_modified_sequence(
            entry.modified_sequence
        )
        aa = self.encode_amino_acids(parsed_sequence)
        mod = self.encode_modifications(parsed_sequence)

        glycan = GlycanNode.from_str(entry.glycan_struct)
        g = self.glycan_to_node_graph(glycan)

        retention_time = torch.tensor(entry.retention_time)
        return GlycoPeptideRTData(
            sequence=aa,
            length=torch.tensor(aa.size(0)),
            modifications=mod,
            glycan_graph=g,
            glycan_position=torch.tensor(entry.glycan_position),
            retention_time=retention_time,
        )


GlycoPeptideRTDataset = RTDataset[GlycoPeptideRTData]


class GlycoPeptideRTOutputConverter(GlycoPeptideRTDataConverter):
    def __init__(
        self,
        sequence_parser: ModifiedSequenceParser,
        monosaccharides: MonosaccharideCollection,
        configs: Union[str, dict, None] = None
    ):
        if configs is None:
            configs = {}
        super().__init__(sequence_parser, monosaccharides, configs)

