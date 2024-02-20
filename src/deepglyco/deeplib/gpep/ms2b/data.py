__all__ = [
    "GlycoPeptideBranchMS2Data",
    "GlycoPeptideBranchMS2Output",
    "GlycoPeptideBranchMS2DataConverter",
    "GlycoPeptideBranchMS2Dataset",
    "GlycoPeptideBranchMS2DataBatch",
]

import itertools
from typing import (
    Any,
    List,
    NamedTuple,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)

import dgl
import numpy as np
import torch

from ..ms2.data import GlycoPeptideMS2Data, GlycoPeptideMS2DataBatch, GlycoPeptideMS2DataConverter, calculate_fragment_intensity_ratio, unbatch_glycan_fragment_intensity

from ....chem.gpep.fragments import (
    GlycanFragmentTypeCollection,
    branch_fragment_graph,
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


class GlycoPeptideBranchMS2Output(NamedTuple):
    peptide_fragment_intensity: torch.Tensor
    glycan_reducing_end_fragment_intensity: torch.Tensor
    glycan_branch_fragment_intensity: torch.Tensor
    fragment_intensity_ratio: torch.Tensor

    def to(self, device):
        return self.__class__(*[x.to(device) for x in self])


class GlycoPeptideBranchMS2Data(GlycoPeptideMS2Data):
    @property
    def glycan_reducing_end_fragment_intensity(self) -> torch.Tensor:
        return self.glycan_fragment_intensity

    @property
    def glycan_branch_fragment_intensity(self) -> torch.Tensor:
        return self.glycan_graph.nodes["branch_composition"].data["fragment_intensity"]


class GlycoPeptideBranchMS2DataBatch(GlycoPeptideMS2DataBatch):
    @property
    def glycan_reducing_end_fragment_intensity(self) -> torch.Tensor:
        return self.glycan_fragment_intensity

    @property
    def glycan_branch_fragment_intensity(self) -> torch.Tensor:
        return self.glycan_graph.nodes["branch_composition"].data["fragment_intensity"]

    @classmethod
    def collate(cls, batch_data: List[GlycoPeptideBranchMS2Data]):
        return super().collate(cast(Any, batch_data))


GlycoPeptideBranchMS2Dataset = MS2Dataset[GlycoPeptideMS2Spectrum, GlycoPeptideBranchMS2Data]


class GlycoPeptideBranchMS2DataConverter(GlycoPeptideMS2DataConverter):
    def __init__(
        self,
        sequence_parser: ModifiedSequenceParser,
        monosaccharides: MonosaccharideCollection,
        peptide_fragment_types: PeptideFragmentTypeCollection,
        glycan_fragment_types: GlycanFragmentTypeCollection,
        configs: Union[str, dict],
    ):
        super().__init__(
            sequence_parser=sequence_parser,
            monosaccharides=monosaccharides,
            peptide_fragment_types=peptide_fragment_types,
            glycan_fragment_types=glycan_fragment_types,
            configs=configs,
        )

    def glycan_to_fragment_graph(self, glycan: GlycanNode):
        fragment_graph = reducing_end_fragment_graph(glycan)
        fragment_graph_b = branch_fragment_graph(fragment_graph)

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
            ("cleavage", "branch_join", "branch_fragment"): (
                [e[0] for e in fragment_graph_b.cleavage_fragment_edges],
                [e[1] for e in fragment_graph_b.cleavage_fragment_edges],
            ),
            ("branch_fragment", "branch_combine", "branch_composition"): (
                [e[0] for e in fragment_graph_b.fragment_composition_edges],
                [e[1] for e in fragment_graph_b.fragment_composition_edges],
            ),
        }

        monosaccharides = self.encode_monosaccharides(
            fragment_graph.monosaccharide_nodes
        )

        g = dgl.convert.heterograph(graph_data)
        g.nodes["monosaccharide"].data["monosaccharide"] = monosaccharides

        return g, fragment_graph, fragment_graph_b


    def glycan_branch_fragment_intensity_to_tensor(
        self,
        composition_nodes: Sequence[MonosaccharideComposition],
        spectrum: GlycoPeptideMS2SpectrumProto,
    ):
        fragment_type = self.get_config("glycan_branch_fragments", "fragment_type", typed=list)
        fragment_charge = self.get_config(
            "glycan_branch_fragments", "fragment_charge", required=False, typed=str
        )
        if fragment_charge != "combined":
            fragment_charge = self.get_config(
                "glycan_branch_fragments", "fragment_charge", typed=list
            )
        fragment_other = self.get_config(
            "glycan_branch_fragments", "other", required=False, typed=str
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
                if self.glycan_fragment_types[ftype].reducing_end:
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
        g, fragment_graph, fragment_graph_b = self.glycan_to_fragment_graph(glycan)

        glycan_fragment_intensity = self.glycan_fragment_intensity_to_tensor(
            fragment_graph.composition_nodes, spectrum
        )
        glycan_branch_fragment_intensity = self.glycan_branch_fragment_intensity_to_tensor(
            fragment_graph_b.composition_nodes, spectrum
        )

        fragment_intensity_ratio = calculate_fragment_intensity_ratio(
            peptide_fragment_intensity, glycan_fragment_intensity
        )

        glycan_branch_fragment_intensity /= glycan_fragment_intensity.max()
        peptide_fragment_intensity = normalize_by_max(peptide_fragment_intensity)
        glycan_fragment_intensity = normalize_by_max(glycan_fragment_intensity)

        g.nodes["composition"].data["fragment_intensity"] = glycan_fragment_intensity
        g.nodes["branch_composition"].data["fragment_intensity"] = glycan_branch_fragment_intensity

        return GlycoPeptideBranchMS2Data(
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
    glycan_reducing_end_fragment_intensity: torch.Tensor,
    glycan_branch_fragment_intensity: torch.Tensor,
    fragment_intensity_ratio: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gly_sum = glycan_reducing_end_fragment_intensity.sum()
    if gly_sum > 0:
        glycan_branch_fragment_intensity = glycan_branch_fragment_intensity / gly_sum
    return (
        normalize_by_sum(peptide_fragment_intensity) * fragment_intensity_ratio,
        normalize_by_sum(glycan_reducing_end_fragment_intensity) * (1 - fragment_intensity_ratio),
        glycan_branch_fragment_intensity * (1 - fragment_intensity_ratio),
    )


@overload
def unbatch_glycan_branch_fragment_intensity(
    batch: GlycoPeptideDataBatch,
    fragment_intensity: torch.Tensor,
    *,
    recover_order: bool = True,
) -> List[torch.Tensor]:
    ...


@overload
def unbatch_glycan_branch_fragment_intensity(
    batch: GlycoPeptideDataBatch,
    fragment_intensity: torch.Tensor,
    *more: torch.Tensor,
    recover_order: bool = True,
) -> List[Tuple[torch.Tensor, ...]]:
    ...


def unbatch_glycan_branch_fragment_intensity(
    batch: GlycoPeptideDataBatch,
    fragment_intensity: torch.Tensor,
    *more: torch.Tensor,
    recover_order: bool = True,
) -> Union[List[torch.Tensor], List[Tuple[torch.Tensor, ...]]]:
    graph = batch.glycan_graph

    result: list[Any] = [None] * batch.batch_size
    with graph.local_scope():
        if len(more) == 0:
            graph.nodes["branch_composition"].data["fragment_intensity"] = fragment_intensity
        else:
            for i, intensity in enumerate(itertools.chain([fragment_intensity], more)):
                graph.nodes["branch_composition"].data[f"fragment_intensity_{i}"] = intensity

        for i, g in enumerate(dgl.unbatch(graph)):
            if recover_order:
                idx = int(batch.indices[i].item())
            else:
                idx = i
            if len(more) == 0:
                result[idx] = g.nodes["branch_composition"].data["fragment_intensity"]
            else:
                result[idx] = tuple(
                    g.nodes["branch_composition"].data[f"fragment_intensity_{i}"]
                    for i in range(len(more) + 1)
                )
        return result


def unbatch_glycopeptide_branch_ms2_output(
    batch: GlycoPeptideDataBatch,
    ms2: GlycoPeptideBranchMS2Output,
    recover_order: bool = True,
) -> List[GlycoPeptideBranchMS2Output]:
    peptide_fragment_intensity = unbatch_peptide_fragment_intensity(
        cast(PeptideDataBatch, batch),
        ms2.peptide_fragment_intensity,
        recover_order=recover_order,
    )
    glycan_fragment_intensity = unbatch_glycan_fragment_intensity(
        batch,
        ms2.glycan_reducing_end_fragment_intensity,
        recover_order=recover_order,
    )
    glycan_branch_fragment_intensity = unbatch_glycan_branch_fragment_intensity(
        batch,
        ms2.glycan_branch_fragment_intensity,
        recover_order=recover_order,
    )
    return [
        GlycoPeptideBranchMS2Output(
            peptide_fragment_intensity[i],
            glycan_fragment_intensity[i],
            glycan_branch_fragment_intensity[i],
            ms2.fragment_intensity_ratio[batch.indices[i] if recover_order else i],
        )
        for i in range(batch.batch_size)
    ]



class GlycoPeptideBranchMS2OutputConverter(GlycoPeptideBranchMS2DataConverter):
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
        ms2: GlycoPeptideBranchMS2Output,
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

        glycan_branch_fragment_type = self.get_config(
            "glycan_branch_fragments", "fragment_type", typed=list
        )
        glycan_branch_fragment_charge = self.get_config(
            "glycan_branch_fragments", "fragment_charge", required=False, typed=str
        )
        if glycan_branch_fragment_charge == "combined":
            glycan_branch_fragment_charge = [1]
        else:
            glycan_branch_fragment_charge = self.get_config(
                "glycan_branch_fragments", "fragment_charge", typed=list
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
            reducing_end_fragment_charge=glycan_fragment_charge,
        )
        glyb_frags = self.mass_calculator.glycan_fragment_mz(
            parsed_sequence=parsed_sequence,
            glycan=glycan,
            fragment_type=glycan_branch_fragment_type,
            branch_fragment_charge=glycan_branch_fragment_charge
        )

        (
            peptide_fragment_intensity,
            glycan_fragment_intensity,
            glycan_branch_fragment_intensity,
        ) = adjust_fragment_intensity_by_ratio(*ms2)
        pep_frag_intensity: np.ndarray = (
            peptide_fragment_intensity.cpu().numpy().flatten(order="F")
        )
        gly_frag_intensity: np.ndarray = (
            glycan_fragment_intensity.cpu().numpy().flatten(order="F")
        )
        glyb_frag_intensity: np.ndarray = (
            glycan_branch_fragment_intensity.cpu().numpy().flatten(order="F")
        )

        assert len(pep_frags.mz) == len(pep_frag_intensity)
        assert len(gly_frags.mz) == len(gly_frag_intensity)
        assert len(glyb_frags.mz) == len(glyb_frag_intensity)

        frag_mz = np.concatenate((pep_frags.mz, gly_frags.mz, glyb_frags.mz))
        frag_intensity = np.concatenate((pep_frag_intensity, gly_frag_intensity, glyb_frag_intensity))
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
            fragment_type=np.concatenate((pep_frags.fragment_type, gly_frags.fragment_type, glyb_frags.fragment_type))[should_keep],
            fragment_number=np.concatenate((pep_frags.fragment_number, gly_frags.fragment_number, glyb_frags.fragment_number))[should_keep],
            loss_type=np.concatenate((pep_frags.loss_type, gly_frags.loss_type, glyb_frags.loss_type))[should_keep],
            fragment_glycan=np.concatenate((pep_frags.fragment_glycan, gly_frags.fragment_glycan, glyb_frags.fragment_glycan))[should_keep],
            fragment_charge=np.concatenate((pep_frags.charge, gly_frags.charge, glyb_frags.charge))[should_keep],
            precursor_mz=float(precursor_mz),
        )
