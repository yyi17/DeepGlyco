__all__ = [
    "PeptideMS2Data",
    "PeptideMS2DataBatch",
    "PeptideMS2DataConverter",
    "PeptideMS2Dataset",
    "unbatch_peptide_fragment_intensity",
    "PeptideMS2OutputConverter",
]

import itertools
from typing import Any, List, NamedTuple, Optional, Tuple, Union, overload

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from ....chem.pep.fragments import PeptideFragmentTypeCollection
from ....chem.pep.mods import ModifiedSequence, ModifiedSequenceParser
from ....chem.pep.pepmass import PeptideMassCalculator
from ....speclib.pep.spec import PeptideMS2Spectrum, PeptideMS2SpectrumProto
from ...common.data import MS2Dataset
from ...util.math import normalize_by_max
from ..common.data import PeptideDataBatch, PeptideDataConverter


class PeptideMS2Data(NamedTuple):
    sequence: torch.Tensor
    length: torch.Tensor
    modifications: Optional[torch.Tensor]
    charge: torch.Tensor
    fragment_intensity: torch.Tensor

    def to(self, device):
        return self.__class__(
            sequence=self.sequence.to(device),
            length=self.length,
            modifications=self.modifications.to(device)
            if self.modifications is not None
            else None,
            charge=self.charge.to(device),
            fragment_intensity=self.fragment_intensity.to(device),
        )


class PeptideMS2DataBatch(NamedTuple):
    sequence: torch.Tensor
    length: torch.Tensor
    modifications: Optional[torch.Tensor]
    charge: torch.Tensor
    fragment_intensity: torch.Tensor
    indices: torch.Tensor

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
            fragment_intensity=self.fragment_intensity.to(device),
            indices=self.indices,
        )

    @classmethod
    def collate(cls, batch_data: List[PeptideMS2Data]):
        length = torch.as_tensor([x.length for x in batch_data])
        indices = torch.argsort(length, descending=True)
        batch_data = [batch_data[i] for i in indices]
        length = length[indices]

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
            fragment_intensity=pad_sequence(
                [x.fragment_intensity for x in batch_data], batch_first=True
            ),
            indices=indices,
        )


PeptideMS2Dataset = MS2Dataset[PeptideMS2Spectrum, PeptideMS2Data]


class PeptideMS2DataConverter(PeptideDataConverter):
    def __init__(
        self,
        sequence_parser: ModifiedSequenceParser,
        peptide_fragment_types: PeptideFragmentTypeCollection,
        configs: Union[str, dict],
    ):
        self.fragment_types = peptide_fragment_types
        super().__init__(sequence_parser, configs)

    def spectrum_to_tensor(self, spectrum: PeptideMS2SpectrumProto) -> PeptideMS2Data:
        parsed_sequence = self.sequence_parser.parse_modified_sequence(
            spectrum.modified_sequence
        )
        aa = self.encode_amino_acids(parsed_sequence)
        mod = self.encode_modifications(parsed_sequence)
        charge = torch.tensor(spectrum.precursor_charge)
        fragment_intensity = self.fragment_intensity_to_tensor(
            parsed_sequence, spectrum
        )
        fragment_intensity = normalize_by_max(fragment_intensity)
        return PeptideMS2Data(
            sequence=aa,
            length=torch.tensor(aa.size(0)),
            modifications=mod,
            charge=charge,
            fragment_intensity=fragment_intensity,
        )

    def fragment_intensity_to_tensor(
        self, parsed_sequence: ModifiedSequence, spectrum: PeptideMS2SpectrumProto
    ) -> torch.Tensor:
        fragment_type = self.get_config("fragments", "fragment_type", typed=list)
        fragment_charge = self.get_config("fragments", "fragment_charge", typed=list)
        loss_type = [""] + (
            self.get_config("fragments", "loss_type", required=False, typed=list) or []
        )

        fragment_intensity = torch.zeros(
            (
                len(parsed_sequence),
                len(fragment_type) * len(loss_type) * len(fragment_charge),
            )
        )

        for i, intensity in enumerate(spectrum.intensity):
            index1 = spectrum.fragment_number[i]
            if index1 <= 0:
                continue
            try:
                if not self.fragment_types[spectrum.fragment_type[i]].n_term:
                    index1 = len(parsed_sequence) - index1

                index2 = (
                    fragment_type.index(spectrum.fragment_type[i])
                    + loss_type.index(spectrum.loss_type[i]) * len(fragment_type)
                    + fragment_charge.index(spectrum.fragment_charge[i])
                    * len(fragment_type)
                    * len(loss_type)
                )
            except (ValueError, KeyError):
                if (
                    self.get_config("fragments", "other", required=False, typed=str)
                    != "ignore"
                ):
                    raise ValueError(
                        f"fragment {spectrum.fragment_type[i]} -{spectrum.loss_type[i]} ({spectrum.fragment_charge[i]}+) not defined"
                    )
                continue
            fragment_intensity[index1 - 1, index2] += torch.tensor(intensity)

        return fragment_intensity


@overload
def unbatch_peptide_fragment_intensity(
    batch: PeptideDataBatch,
    fragment_intensity: torch.Tensor,
    *,
    recover_order: bool = True,
) -> List[torch.Tensor]:
    ...


@overload
def unbatch_peptide_fragment_intensity(
    batch: PeptideDataBatch,
    fragment_intensity: torch.Tensor,
    *more: torch.Tensor,
    recover_order: bool = True,
) -> List[Tuple[torch.Tensor, ...]]:
    ...


def unbatch_peptide_fragment_intensity(
    batch: PeptideDataBatch,
    fragment_intensity: torch.Tensor,
    *more: torch.Tensor,
    recover_order: bool = True,
) -> Union[List[torch.Tensor], List[Tuple[torch.Tensor, ...]]]:
    result: list[Any] = [None] * batch.batch_size
    for i in range(batch.batch_size):
        if recover_order:
            idx = int(batch.indices[i].item())
        else:
            idx = i
        seq_len = batch.length[i]
        if len(more) == 0:
            result[idx] = fragment_intensity[i][: (seq_len - 1)]
        else:
            result[idx] = tuple(
                intensity[i][: (seq_len - 1)]
                for intensity in itertools.chain([fragment_intensity], more)
            )
    return result


class PeptideMS2OutputConverter(PeptideDataConverter):
    def __init__(
        self,
        sequence_parser: ModifiedSequenceParser,
        mass_calculator: PeptideMassCalculator,
        peptide_fragment_types: PeptideFragmentTypeCollection,
        configs: Union[str, dict, None] = None,
    ):
        if configs is None:
            configs = {}

        self.mass_calculator = mass_calculator
        self.fragment_types = peptide_fragment_types
        super().__init__(sequence_parser, configs)

    def tensor_to_spectrum(
        self,
        sequence: str,
        precursor_charge: int,
        fragment_intensity: torch.Tensor,
        keep_zeros: bool = False,
    ):
        fragment_type = self.get_config("fragments", "fragment_type", typed=list)
        fragment_charge = self.get_config("fragments", "fragment_charge", typed=list)
        loss_type = [""] + (
            self.get_config("fragments", "loss_type", required=False, typed=list) or []
        )

        parsed_sequence = self.sequence_parser.parse_modified_sequence(sequence)
        precursor_mz = self.mass_calculator.precursor_mz(
            parsed_sequence=parsed_sequence, charge=precursor_charge
        )
        fragments = self.mass_calculator.fragment_mz(
            parsed_sequence=parsed_sequence,
            fragment_type=fragment_type,
            loss_type=loss_type,
            charge=fragment_charge,
            keep_fragment_placeholder=True,
        )

        frag_intensity: np.ndarray = normalize_by_max(fragment_intensity).cpu().numpy()
        frag_intensity = frag_intensity.flatten(order="F")

        assert len(fragments.mz) == len(frag_intensity)

        should_keep = ~np.isnan(fragments.mz)
        if not keep_zeros:
            should_keep &= frag_intensity > 0

        return PeptideMS2Spectrum(
            modified_sequence=sequence,
            precursor_charge=precursor_charge,
            mz=fragments.mz[should_keep],
            intensity=frag_intensity[should_keep],
            fragment_type=fragments.fragment_type[should_keep],
            fragment_number=fragments.fragment_number[should_keep],
            loss_type=fragments.loss_type[should_keep],
            fragment_charge=fragments.charge[should_keep],
            precursor_mz=float(precursor_mz),
        )
