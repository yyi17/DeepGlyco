__all__ = ["PeptideData", "PeptideDataConverter", "PeptideDataBatch"]

from typing import List, NamedTuple, Optional, Union

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ....chem.pep.mods import ModifiedSequence, ModifiedSequenceParser
from ....util.config import Configurable


class PeptideData(NamedTuple):
    sequence: torch.Tensor
    length: torch.Tensor
    modifications: Optional[torch.Tensor]
    charge: Optional[torch.Tensor]

    def to(self, device):
        return self.__class__(
            sequence=self.sequence.to(device),
            length=self.length,
            modifications=self.modifications.to(device)
            if self.modifications is not None
            else None,
            charge=self.charge.to(device) if self.charge is not None else None,
        )


class PeptideDataBatch(NamedTuple):
    sequence: torch.Tensor
    length: torch.Tensor
    modifications: Optional[torch.Tensor]
    charge: Optional[torch.Tensor]
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
            charge=self.charge.to(device) if self.charge is not None else None,
            indices=self.indices,
        )

    @classmethod
    def collate(cls, batch_data: List[PeptideData]):
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
            raise ValueError(f"modifications contain None")

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
            charge=torch.as_tensor(charge) if charge is not None else None,
            indices=indices,
        )


class PeptideDataConverter(Configurable):
    def __init__(
        self,
        sequence_parser: ModifiedSequenceParser,
        configs: Union[str, dict],
    ):
        self.sequence_parser = sequence_parser
        super().__init__(configs)

    def peptide_to_tensor(
        self, sequence: str, precursor_charge: Optional[int] = None
    ) -> PeptideData:
        parsed_sequence = self.sequence_parser.parse_modified_sequence(sequence)
        aa = self.encode_amino_acids(parsed_sequence)
        mod = self.encode_modifications(parsed_sequence)
        if precursor_charge is None:
            charge = None
        else:
            charge = torch.tensor(precursor_charge)
        return PeptideData(
            sequence=aa,
            length=torch.tensor(aa.size(0)),
            modifications=mod,
            charge=charge,
        )

    def encode_amino_acids(self, parsed_sequence: ModifiedSequence) -> torch.Tensor:
        amino_acids = self.get_config("amino_acids", "symbol", typed=list)
        aa_indices = []
        for aa, _ in parsed_sequence:
            try:
                aa_indices.append(amino_acids.index(aa) + 1)
            except ValueError:
                raise ValueError(f"amino acid {aa} not defined")

        aa_indices = torch.tensor(aa_indices)
        if (
            self.get_config("amino_acids", "encoding", required=False, typed=str)
            == "onehot"
        ):
            x = F.one_hot(aa_indices, len(amino_acids) + 1)
            return x[:, 1:]
        return aa_indices

    def encode_modifications(
        self, parsed_sequence: ModifiedSequence
    ) -> Optional[torch.Tensor]:
        if self.get_config("modifications", required=False, typed=dict) is None:
            return None

        fixed_modifications = (
            self.get_config("modifications", "fixed", required=False, typed=list) or []
        )
        encoding = self.get_config(
            "modifications", "encoding", required=False, typed=str
        )
        if encoding == "atoms":
            atoms = self.get_config("modifications", "element", typed=list)
            mod_atoms = []
            for _, mod in parsed_sequence:
                atom_counts = [0] * len(atoms)
                if mod != "" and mod not in fixed_modifications:
                    composition = self.sequence_parser.modifications[mod].composition
                    if composition is None:
                        raise ValueError(
                            f"atom composition of modification {mod} not defined"
                        )
                    for atom, num in composition.items():
                        try:
                            atom_counts[atoms.index(atom)] = num
                        except ValueError:
                            raise ValueError(
                                f"element {atom} in modification {mod} not defined"
                            )
                mod_atoms.append(atom_counts)
            return torch.tensor(mod_atoms)
        else:
            mods = self.get_config("modifications", "symbol", typed=list)
            mod_indices = []

            for _, mod in parsed_sequence:
                if mod == "" or mod in fixed_modifications:
                    mod_indices.append(0)
                    continue
                mod = "".join(ch for ch in mod if ch.islower() and ch.isalnum())
                try:
                    mod_indices.append(mods.index(mod) + 1)
                except ValueError:
                    raise ValueError(f"modification {mod} not defined")
            mod_indices = torch.tensor(mod_indices)

            if encoding == "onehot":
                x = F.one_hot(mod_indices, len(mods) + 1)
                return x[:, 1:]
            return mod_indices
