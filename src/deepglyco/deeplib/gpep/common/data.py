__all__ = ["GlycoPeptideData", "GlycoPeptideDataConverter", "GlycoPeptideDataBatch"]

from typing import List, NamedTuple, Optional, Sequence, Union

import dgl
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ....chem.gpep.glycans import (
    GlycanNode,
    MonosaccharideCollection,
    glycan_node_graph,
)
from ....chem.pep.mods import ModifiedSequenceParser
from ...pep.ms2.data import PeptideDataConverter


class GlycoPeptideData(NamedTuple):
    sequence: torch.Tensor
    length: torch.Tensor
    modifications: Optional[torch.Tensor]
    charge: Optional[torch.Tensor]
    glycan_graph: dgl.DGLGraph
    glycan_position: torch.Tensor

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
            charge=self.charge.to(device) if self.charge is not None else None,
            glycan_graph=self.glycan_graph.to(device),
            glycan_position=self.glycan_position.to(device),
        )


class GlycoPeptideDataBatch(NamedTuple):
    sequence: torch.Tensor
    length: torch.Tensor
    modifications: Optional[torch.Tensor]
    charge: Optional[torch.Tensor]
    glycan_graph: dgl.DGLGraph
    glycan_position: torch.Tensor
    indices: torch.Tensor

    @property
    def batch_size(self):
        return self.indices.size(0)

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
            charge=self.charge.to(device) if self.charge is not None else None,
            glycan_graph=self.glycan_graph.to(device),
            glycan_position=self.glycan_position.to(device),
            indices=self.indices,
        )

    @classmethod
    def collate(cls, batch_data: List[GlycoPeptideData]):
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

        charge = [x.charge for x in batch_data if x.charge is not None]
        if len(charge) == 0:
            charge = None
        elif len(charge) != len(batch_data):
            raise ValueError(f"charge contains None")

        return cls(
            sequence=pad_sequence([x.sequence for x in batch_data], batch_first=True),
            length=length,
            modifications=pad_sequence(modifications, batch_first=True)
            if modifications is not None
            else None,
            charge=torch.as_tensor([x.charge for x in batch_data])
            if charge is not None
            else None,
            glycan_graph=graph,
            glycan_position=torch.as_tensor([x.glycan_position for x in batch_data]),
            indices=indices,
        )


class GlycoPeptideDataConverter(PeptideDataConverter):
    def __init__(
        self,
        sequence_parser: ModifiedSequenceParser,
        monosaccharides: MonosaccharideCollection,
        configs: Union[str, dict],
    ):
        super().__init__(
            sequence_parser=sequence_parser,
            configs=configs,
        )
        self.monosaccharides = monosaccharides

    def encode_monosaccharides(self, nodes: Sequence[GlycanNode]):
        monosaccharides = self.get_config("monosaccharides", "symbol", typed=list)
        indices = []
        for n in nodes:
            try:
                indices.append(monosaccharides.index(n.monosaccharide) + 1)
            except ValueError:
                raise ValueError(f"monosaccharide {n.monosaccharide} not defined")
        indices = torch.tensor(indices)

        if (
            self.get_config("monosaccharides", "encoding", required=False, typed=str)
            == "onehot"
        ):
            x = F.one_hot(indices, len(monosaccharides) + 1)
            return x[:, 1:]
        return indices

    def glycan_to_node_graph(self, glycan: GlycanNode):
        node_graph = glycan_node_graph(glycan)
        graph_data = {
            ("monosaccharide", "link", "monosaccharide"): (
                [e[0] for e in node_graph.edges],
                [e[1] for e in node_graph.edges],
            ),
        }
        g = dgl.convert.heterograph(graph_data)

        monosaccharides = self.encode_monosaccharides(node_graph.nodes)
        g.nodes["monosaccharide"].data["monosaccharide"] = monosaccharides
        return g

    def glycopeptide_to_tensor(
        self,
        sequence: str,
        glycan_struct: str,
        glycan_position: int,
        precursor_charge: Optional[int] = None,
    ) -> GlycoPeptideData:
        parsed_sequence = self.sequence_parser.parse_modified_sequence(sequence)
        aa = self.encode_amino_acids(parsed_sequence)
        mod = self.encode_modifications(parsed_sequence)
        if precursor_charge is None:
            charge = None
        else:
            charge = torch.tensor(precursor_charge)

        glycan = GlycanNode.from_str(glycan_struct)
        g = self.glycan_to_node_graph(glycan)

        return GlycoPeptideData(
            sequence=aa,
            length=torch.tensor(aa.size(0)),
            modifications=mod,
            charge=charge,
            glycan_graph=g,
            glycan_position=torch.tensor(glycan_position),
        )
