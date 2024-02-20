__all__ = [
    "GlycoPeptideMS2Data",
    "GlycoPeptideMS2Output",
    "GlycoPeptideMS2DataConverter",
    "GlycoPeptideMS2Dataset",
    "GlycoPeptideMS2DataBatch",
]

import itertools
from typing import (
    Any,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)

import dgl
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from ....chem.gpep.fragments import (
    GlycanFragmentTypeCollection,
    reducing_end_fragment_graph,
)
from ....chem.gpep.glycans import (
    GlycanNode,
    MonosaccharideCollection,
    MonosaccharideComposition,
)
from ....chem.gpep.gpmass import GlycoPeptideMassCalculator
from ....chem.pep.fragments import PeptideFragmentTypeCollection
from ....chem.pep.mods import ModifiedSequence, ModifiedSequenceParser
from ....speclib.gpep.spec import GlycoPeptideMS2Spectrum, GlycoPeptideMS2SpectrumProto
from ....util.math import normalize_by_max as np_normalize_by_max
from ...common.data import MS2Dataset
from ...pep.common.data import PeptideDataBatch
from ...pep.ms2.data import unbatch_peptide_fragment_intensity
from ...util.math import normalize_by_max, normalize_by_sum
from ..common.data import (
    GlycoPeptideData,
    GlycoPeptideDataBatch,
    GlycoPeptideDataConverter,
)


class GlycoPeptideMS2Output(NamedTuple):
    peptide_fragment_intensity: torch.Tensor
    glycan_fragment_intensity: torch.Tensor
    fragment_intensity_ratio: torch.Tensor

    def to(self, device):
        return self.__class__(*[x.to(device) for x in self])


class GlycoPeptideMS2Data(NamedTuple):
    sequence: torch.Tensor
    length: torch.Tensor
    modifications: Optional[torch.Tensor]
    charge: torch.Tensor
    glycan_graph: dgl.DGLGraph
    glycan_position: torch.Tensor
    peptide_fragment_intensity: torch.Tensor
    fragment_intensity_ratio: torch.Tensor

    @property
    def monosaccharides(self):
        return self.glycan_graph.nodes["monosaccharide"].data["monosaccharide"]

    @property
    def glycan_fragment_intensity(self) -> torch.Tensor:
        return self.glycan_graph.nodes["composition"].data["fragment_intensity"]

    def to(self, device):
        return self.__class__(
            sequence=self.sequence.to(device),
            length=self.length,
            modifications=self.modifications.to(device)
            if self.modifications is not None
            else None,
            charge=self.charge.to(device),
            glycan_graph=self.glycan_graph.to(device),
            glycan_position=self.glycan_position.to(device),
            peptide_fragment_intensity=self.peptide_fragment_intensity.to(device),
            fragment_intensity_ratio=self.fragment_intensity_ratio.to(device),
        )


class GlycoPeptideMS2DataBatch(NamedTuple):
    sequence: torch.Tensor
    length: torch.Tensor
    modifications: Optional[torch.Tensor]
    charge: torch.Tensor
    glycan_graph: dgl.DGLGraph
    glycan_position: torch.Tensor
    peptide_fragment_intensity: torch.Tensor
    fragment_intensity_ratio: torch.Tensor
    indices: torch.Tensor

    @property
    def monosaccharides(self):
        return self.glycan_graph.nodes["monosaccharide"].data["monosaccharide"]

    @property
    def glycan_fragment_intensity(self) -> torch.Tensor:
        return self.glycan_graph.nodes["composition"].data["fragment_intensity"]

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
            charge=self.charge.to(device),
            glycan_graph=self.glycan_graph.to(device),
            glycan_position=self.glycan_position.to(device),
            peptide_fragment_intensity=self.peptide_fragment_intensity.to(device),
            fragment_intensity_ratio=self.fragment_intensity_ratio.to(device),
            indices=self.indices,
        )

    @classmethod
    def collate(cls, batch_data: List[GlycoPeptideMS2Data]):
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
            charge=torch.as_tensor([x.charge for x in batch_data]),
            glycan_graph=graph,
            glycan_position=torch.as_tensor([x.glycan_position for x in batch_data]),
            peptide_fragment_intensity=pad_sequence(
                [x.peptide_fragment_intensity for x in batch_data], batch_first=True
            ),
            fragment_intensity_ratio=torch.as_tensor(
                [x.fragment_intensity_ratio for x in batch_data]
            ),
            indices=indices,
        )


GlycoPeptideMS2Dataset = MS2Dataset[GlycoPeptideMS2Spectrum, GlycoPeptideMS2Data]


def calculate_fragment_intensity_ratio(
    peptide_fragment_intensity: torch.Tensor, glycan_fragment_intensity: torch.Tensor
) -> torch.Tensor:
    peptide_fragment_intensity_sum = peptide_fragment_intensity.sum()
    glycan_fragment_intensity_sum = glycan_fragment_intensity.sum()
    if peptide_fragment_intensity_sum + glycan_fragment_intensity_sum == 0:
        fragment_intensity_ratio = torch.tensor(1.0)
    else:
        fragment_intensity_ratio = peptide_fragment_intensity_sum / (
            peptide_fragment_intensity_sum + glycan_fragment_intensity_sum
        )
    return fragment_intensity_ratio


class GlycoPeptideMS2DataConverter(GlycoPeptideDataConverter):
    def __init__(
        self,
        sequence_parser: ModifiedSequenceParser,
        monosaccharides: MonosaccharideCollection,
        peptide_fragment_types: PeptideFragmentTypeCollection,
        glycan_fragment_types: GlycanFragmentTypeCollection,
        configs: Union[str, dict],
    ):
        self.peptide_fragment_types = peptide_fragment_types
        self.glycan_fragment_types = glycan_fragment_types
        super().__init__(
            sequence_parser=sequence_parser,
            monosaccharides=monosaccharides,
            configs=configs,
        )

    def glycan_to_fragment_graph(self, glycan: GlycanNode):
        fragment_graph = reducing_end_fragment_graph(glycan)

        graph_data: dict = {
            ("monosaccharide", "link", "monosaccharide"): (
                [e[0] for e in fragment_graph.monosaccharide_edges],
                [e[1] for e in fragment_graph.monosaccharide_edges],
            ),
            ("monosaccharide", "retained", "cleavage"): (
                [e[0] for e in fragment_graph.retained_monosaccharide_cleavage_edges],
                [e[1] for e in fragment_graph.retained_monosaccharide_cleavage_edges],
            ),
            ("monosaccharide", "lost", "cleavage"): (
                [e[0] for e in fragment_graph.lost_monosaccharide_cleavage_edges],
                [e[1] for e in fragment_graph.lost_monosaccharide_cleavage_edges],
            ),
            ("cleavage", "join", "fragment"): (
                [e[0] for e in fragment_graph.cleavage_fragment_edges],
                [e[1] for e in fragment_graph.cleavage_fragment_edges],
            ),
            ("fragment", "combine", "composition"): (
                [e[0] for e in fragment_graph.fragment_composition_edges],
                [e[1] for e in fragment_graph.fragment_composition_edges],
            ),
        }

        monosaccharides = self.encode_monosaccharides(
            fragment_graph.monosaccharide_nodes
        )

        g = dgl.convert.heterograph(graph_data)
        g.nodes["monosaccharide"].data["monosaccharide"] = monosaccharides

        return g, fragment_graph

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
        g = self.glycan_to_fragment_graph(glycan)[0]

        return GlycoPeptideData(
            sequence=aa,
            length=torch.tensor(aa.size(0)),
            modifications=mod,
            charge=charge,
            glycan_graph=g,
            glycan_position=torch.tensor(glycan_position),
        )

    def peptide_fragment_intensity_to_tensor(
        self, parsed_sequence: ModifiedSequence, spectrum: GlycoPeptideMS2SpectrumProto
    ):
        fragment_type = self.get_config(
            "peptide_fragments", "fragment_type", typed=list
        )
        fragment_charge = self.get_config(
            "peptide_fragments", "fragment_charge", required=False, typed=str
        )
        if fragment_charge != "combined":
            fragment_charge = self.get_config(
                "peptide_fragments", "fragment_charge", typed=list
            )
        loss_type = [""] + (
            self.get_config(
                "peptide_fragments", "loss_type", required=False, typed=list
            )
            or []
        )
        fragment_glycan = [""] + (
            self.get_config(
                "peptide_fragments", "fragment_glycan", required=False, typed=list
            )
            or []
        )

        num_fragment_types = len(fragment_type) * len(loss_type) * len(fragment_glycan)
        if fragment_charge != "combined":
            num_fragment_types *= len(fragment_charge)
        fragment_intensity = torch.zeros((len(parsed_sequence), num_fragment_types))
        for i, intensity in enumerate(spectrum.intensity):
            index1 = spectrum.fragment_number[i]
            if index1 <= 0:
                continue
            try:
                if not self.peptide_fragment_types[spectrum.fragment_type[i]].n_term:
                    index1 = len(parsed_sequence) - index1

                index2 = (
                    fragment_type.index(spectrum.fragment_type[i])
                    + loss_type.index(spectrum.loss_type[i]) * len(fragment_type)
                    + fragment_glycan.index(spectrum.fragment_glycan[i])
                    * len(fragment_type)
                    * len(loss_type)
                )
                if fragment_charge != "combined":
                    index2 += (
                        fragment_charge.index(spectrum.fragment_charge[i])
                        * len(fragment_type)
                        * len(loss_type)
                        * len(fragment_glycan)
                    )
            except (ValueError, KeyError):
                if (
                    self.get_config(
                        "peptide_fragments", "other", required=False, typed=str
                    )
                    != "ignore"
                ):
                    raise ValueError(
                        f"peptide fragment {spectrum.fragment_type[i]}-{spectrum.fragment_glycan[i]} -{spectrum.loss_type[i]}  ({spectrum.fragment_charge[i]}+) not defined"
                    )
                continue
            fragment_intensity[index1 - 1, index2] += torch.tensor(intensity)
        return fragment_intensity

    def glycan_fragment_intensity_to_tensor(
        self,
        composition_nodes: Sequence[MonosaccharideComposition],
        spectrum: GlycoPeptideMS2SpectrumProto,
    ):
        fragment_type = self.get_config("glycan_fragments", "fragment_type", typed=list)
        fragment_charge = self.get_config(
            "glycan_fragments", "fragment_charge", required=False, typed=str
        )
        if fragment_charge != "combined":
            fragment_charge = self.get_config(
                "glycan_fragments", "fragment_charge", typed=list
            )
        fragment_other = self.get_config(
            "glycan_fragments", "other", required=False, typed=str
        )

        num_fragment_types = len(fragment_type)
        if fragment_charge != "combined":
            num_fragment_types *= len(fragment_charge)
        fragment_intensity = torch.zeros((len(composition_nodes), num_fragment_types))
        for i, intensity in enumerate(spectrum.intensity):
            index1 = spectrum.fragment_number[i]
            if index1 > 0:
                continue
            ftype = spectrum.fragment_type[i]

            try:
                if not self.glycan_fragment_types[ftype].reducing_end:
                    continue
                index2 = fragment_type.index(ftype)
                if fragment_charge != "combined":
                    index2 += fragment_charge.index(spectrum.fragment_charge[i]) * len(
                        fragment_type
                    )
            except (ValueError, KeyError):
                if fragment_other != "ignore":
                    raise ValueError(
                        f"glycan fragment {spectrum.fragment_type[i]} ({spectrum.fragment_charge[i]}+) not defined"
                    )
                continue
            try:
                if spectrum.fragment_glycan[i] == "":
                    comp = {}
                else:
                    comp = self.monosaccharides.parse_monosaccharide_composition(
                        spectrum.fragment_glycan[i]
                    )
                index1 = composition_nodes.index(comp)
            except ValueError:
                if fragment_other != "ignore":
                    raise ValueError(
                        f"glycan fragment {spectrum.fragment_type[i]} {spectrum.fragment_glycan[i]} ({spectrum.fragment_charge[i]}+) is unknown"
                    )
                continue
            fragment_intensity[index1, index2] += torch.tensor(intensity)

        return fragment_intensity

    def spectrum_to_tensor(self, spectrum: GlycoPeptideMS2Spectrum):
        parsed_sequence = self.sequence_parser.parse_modified_sequence(
            spectrum.modified_sequence
        )
        aa = self.encode_amino_acids(parsed_sequence)
        mod = self.encode_modifications(parsed_sequence)
        charge = torch.tensor(spectrum.precursor_charge)
        peptide_fragment_intensity = self.peptide_fragment_intensity_to_tensor(
            parsed_sequence, spectrum
        )

        glycan = GlycanNode.from_str(spectrum.glycan_struct)
        g, fragment_graph = self.glycan_to_fragment_graph(glycan)

        glycan_fragment_intensity = self.glycan_fragment_intensity_to_tensor(
            fragment_graph.composition_nodes, spectrum
        )

        fragment_intensity_ratio = calculate_fragment_intensity_ratio(
            peptide_fragment_intensity, glycan_fragment_intensity
        )

        peptide_fragment_intensity = normalize_by_max(peptide_fragment_intensity)
        glycan_fragment_intensity = normalize_by_max(glycan_fragment_intensity)
        g.nodes["composition"].data["fragment_intensity"] = glycan_fragment_intensity

        return GlycoPeptideMS2Data(
            sequence=aa,
            length=torch.tensor(aa.size(0)),
            modifications=mod,
            charge=charge,
            glycan_graph=g,
            glycan_position=torch.tensor(spectrum.glycan_position),
            peptide_fragment_intensity=peptide_fragment_intensity,
            fragment_intensity_ratio=fragment_intensity_ratio,
        )


def adjust_fragment_intensity_by_ratio(
    peptide_fragment_intensity: torch.Tensor,
    glycan_fragment_intensity: torch.Tensor,
    fragment_intensity_ratio: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return (
        normalize_by_sum(peptide_fragment_intensity) * fragment_intensity_ratio,
        normalize_by_sum(glycan_fragment_intensity) * (1 - fragment_intensity_ratio),
    )


@overload
def unbatch_glycan_fragment_intensity(
    batch: GlycoPeptideDataBatch,
    fragment_intensity: torch.Tensor,
    *,
    recover_order: bool = True,
) -> List[torch.Tensor]:
    ...


@overload
def unbatch_glycan_fragment_intensity(
    batch: GlycoPeptideDataBatch,
    fragment_intensity: torch.Tensor,
    *more: torch.Tensor,
    recover_order: bool = True,
) -> List[Tuple[torch.Tensor, ...]]:
    ...


def unbatch_glycan_fragment_intensity(
    batch: GlycoPeptideDataBatch,
    fragment_intensity: torch.Tensor,
    *more: torch.Tensor,
    recover_order: bool = True,
) -> Union[List[torch.Tensor], List[Tuple[torch.Tensor, ...]]]:
    graph = batch.glycan_graph

    result: list[Any] = [None] * batch.batch_size
    with graph.local_scope():
        if len(more) == 0:
            graph.nodes["composition"].data["fragment_intensity"] = fragment_intensity
        else:
            for i, intensity in enumerate(itertools.chain([fragment_intensity], more)):
                graph.nodes["composition"].data[f"fragment_intensity_{i}"] = intensity

        for i, g in enumerate(dgl.unbatch(graph)):
            if recover_order:
                idx = int(batch.indices[i].item())
            else:
                idx = i
            if len(more) == 0:
                result[idx] = g.nodes["composition"].data["fragment_intensity"]
            else:
                result[idx] = tuple(
                    g.nodes["composition"].data[f"fragment_intensity_{i}"]
                    for i in range(len(more) + 1)
                )
        return result


def unbatch_glycopeptide_ms2_output(
    batch: GlycoPeptideDataBatch,
    ms2: GlycoPeptideMS2Output,
    recover_order: bool = True,
) -> List[GlycoPeptideMS2Output]:
    peptide_fragment_intensity = unbatch_peptide_fragment_intensity(
        cast(PeptideDataBatch, batch),
        ms2.peptide_fragment_intensity,
        recover_order=recover_order,
    )
    glycan_fragment_intensity = unbatch_glycan_fragment_intensity(
        batch,
        ms2.glycan_fragment_intensity,
        recover_order=recover_order,
    )
    return [
        GlycoPeptideMS2Output(
            peptide_fragment_intensity[i],
            glycan_fragment_intensity[i],
            ms2.fragment_intensity_ratio[batch.indices[i] if recover_order else i],
        )
        for i in range(batch.batch_size)
    ]


class GlycoPeptideMS2OutputConverter(GlycoPeptideMS2DataConverter):
    def __init__(
        self,
        sequence_parser: ModifiedSequenceParser,
        mass_calculator: GlycoPeptideMassCalculator,
        monosaccharides: MonosaccharideCollection,
        peptide_fragment_types: PeptideFragmentTypeCollection,
        glycan_fragment_types: GlycanFragmentTypeCollection,
        configs: Union[str, dict, None] = None,
    ):
        if configs is None:
            configs = {}

        self.mass_calculator = mass_calculator
        super().__init__(
            sequence_parser=sequence_parser,
            monosaccharides=monosaccharides,
            peptide_fragment_types=peptide_fragment_types,
            glycan_fragment_types=glycan_fragment_types,
            configs=configs,
        )

    def tensor_to_spectrum(
        self,
        sequence: str,
        glycan_struct: str,
        glycan_position: int,
        precursor_charge: int,
        ms2: GlycoPeptideMS2Output,
        keep_zeros: bool = False,
    ):
        peptide_fragment_type = self.get_config(
            "peptide_fragments", "fragment_type", typed=list
        )
        peptide_fragment_charge = self.get_config(
            "peptide_fragments", "fragment_charge", required=False, typed=str
        )
        if peptide_fragment_charge == "combined":
            peptide_fragment_charge = [1]
        else:
            peptide_fragment_charge = self.get_config(
                "peptide_fragments", "fragment_charge", typed=list
            )
        peptide_fragment_loss_type = [""] + (
            self.get_config(
                "peptide_fragments", "loss_type", required=False, typed=list
            )
            or []
        )
        peptide_fragment_glycan = [""] + (
            self.get_config(
                "peptide_fragments", "fragment_glycan", required=False, typed=list
            )
            or []
        )

        glycan_fragment_type = self.get_config(
            "glycan_fragments", "fragment_type", typed=list
        )
        glycan_fragment_charge = self.get_config(
            "glycan_fragments", "fragment_charge", required=False, typed=str
        )
        if glycan_fragment_charge == "combined":
            glycan_fragment_charge = [2]
        else:
            glycan_fragment_charge = self.get_config(
                "glycan_fragments", "fragment_charge", typed=list
            )

        parsed_sequence = self.sequence_parser.parse_modified_sequence(sequence)
        glycan = GlycanNode.from_str(glycan_struct)

        precursor_mz = self.mass_calculator.precursor_mz(
            parsed_sequence=parsed_sequence, glycan=glycan, charge=precursor_charge
        )
        pep_frags = self.mass_calculator.peptide_fragment_mz(
            parsed_sequence=parsed_sequence,
            glycan_position=glycan_position,
            fragment_type=peptide_fragment_type,
            loss_type=peptide_fragment_loss_type,
            fragment_glycan=peptide_fragment_glycan,
            fragment_charge=peptide_fragment_charge,
            keep_fragment_placeholder=True,
        )
        gly_frags = self.mass_calculator.glycan_fragment_mz(
            parsed_sequence=parsed_sequence,
            glycan=glycan,
            fragment_type=glycan_fragment_type,
            fragment_charge=glycan_fragment_charge,
        )

        (
            peptide_fragment_intensity,
            glycan_fragment_intensity,
        ) = adjust_fragment_intensity_by_ratio(*ms2)
        pep_frag_intensity: np.ndarray = (
            peptide_fragment_intensity.cpu().numpy().flatten(order="F")
        )
        gly_frag_intensity: np.ndarray = (
            glycan_fragment_intensity.cpu().numpy().flatten(order="F")
        )

        assert len(pep_frags.mz) == len(pep_frag_intensity)
        assert len(gly_frags.mz) == len(gly_frag_intensity)

        frag_mz = np.concatenate((pep_frags.mz, gly_frags.mz))
        frag_intensity = np.concatenate((pep_frag_intensity, gly_frag_intensity))
        frag_intensity = np_normalize_by_max(frag_intensity)

        should_keep = ~np.isnan(frag_mz)
        if not keep_zeros:
            should_keep &= frag_intensity > 0

        return GlycoPeptideMS2Spectrum(
            modified_sequence=sequence,
            glycan_struct=glycan_struct,
            glycan_position=glycan_position,
            precursor_charge=precursor_charge,
            mz=frag_mz[should_keep],
            intensity=frag_intensity[should_keep],
            fragment_type=np.concatenate((pep_frags.fragment_type, gly_frags.fragment_type))[should_keep],
            fragment_number=np.concatenate((pep_frags.fragment_number, gly_frags.fragment_number))[should_keep],
            loss_type=np.concatenate((pep_frags.loss_type, gly_frags.loss_type))[should_keep],
            fragment_glycan=np.concatenate((pep_frags.fragment_glycan, gly_frags.fragment_glycan))[should_keep],
            fragment_charge=np.concatenate((pep_frags.charge, gly_frags.charge))[should_keep],
            precursor_mz=float(precursor_mz),
        )
