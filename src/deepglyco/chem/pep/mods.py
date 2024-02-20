__all__ = [
    "ModPosition",
    "ModificationInfo",
    "ModificationCollection",
    "ModifiedSequence",
    "modified_sequence_to_str",
    "ModifiedSequenceParser",
    "ModifiedSequenceFormatterUnimod",
    "ModifiedSequenceFormatterTPP",
]

import itertools
import os
import re
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Mapping, Optional, Tuple

from ...util.io.yaml import load_yaml
from ..common.elements import ElementCollection, ElementComposition
from ..pep.aminoacids import AminoAcidCollection


class ModPosition(IntEnum):
    none = 0b0000000

    not_term = 0b0000001
    protein_n_term = 0b0000010
    protein_c_term = 0b0000100
    protein_term = 0b0000110

    nonprotein_n_term = 0b0001000
    nonprotein_c_term = 0b0010000
    nonprotein_term = 0b0011000

    any_n_term = 0b0001010
    any_c_term = 0b0010100
    any_term = 0b0011110

    n_term = any_n_term
    c_term = any_c_term

    not_n_term = 0b0010101
    not_c_term = 0b0001011
    anywhere = 0b0011111


@dataclass
class ModificationInfo:
    name: str
    mass: float
    amino_acid: str = ""
    position: ModPosition = ModPosition.anywhere
    loss: str = ""
    unimod: int = -1
    composition: Optional[ElementComposition] = None


class ModificationCollection(Dict[str, ModificationInfo]):
    def __init__(self, modification_map: Mapping[str, ModificationInfo]):
        super().__init__(**modification_map)
        for key, mod in modification_map.items():
            self.check_mod_id(key, mod)

        amino_acid_map = {}
        for key, mod in self.items():
            if mod.amino_acid not in amino_acid_map:
                amino_acid_map[mod.amino_acid] = [key]
            else:
                amino_acid_map[mod.amino_acid].append(key)
        self._amino_acid_map = amino_acid_map

        position_map = {}
        for key, mod in self.items():
            if mod.position not in position_map:
                position_map[mod.position] = [key]
            else:
                position_map[mod.position].append(key)
        self._position_map = position_map

        unimod_map = {}
        for key, mod in self.items():
            if mod.unimod not in unimod_map:
                unimod_map[mod.unimod] = [key]
            else:
                unimod_map[mod.unimod].append(key)
        self._unimod_map = unimod_map

    @classmethod
    def load(cls, elements: ElementCollection, modification_file: Optional[str] = None):
        if modification_file is None:
            modification_file = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "modifications.yaml"
            )

        dict_ = load_yaml(modification_file)

        def from_dict(d: Dict):
            composition = d.get("composition", None)
            if isinstance(composition, str):
                composition = elements.parse_element_composition(composition)

            mass = d.get("mass", None)
            if mass is None and composition:
                mass = elements.mass_from_element_composition(composition)
            if mass is None:
                raise ValueError(f"mass is missing")
            position = ModPosition[d.get("position", "anywhere")]

            return ModificationInfo(
                name=d["name"],
                mass=mass,
                composition=composition,
                amino_acid=d.get("amino_acid", ""),
                position=position,
                loss=d.get("loss", ""),
                unimod=d.get("unimod", -1),
            )

        collection = cls({key: from_dict(value) for key, value in dict_.items()})
        return collection

    def check_mod_id(self, id: str, mod_info: ModificationInfo):
        if not id:
            raise ValueError(f"invalid modification id '{id}': empty")
        if mod_info.position == ModPosition.anywhere:
            suffix = mod_info.amino_acid or "*"
        else:
            suffix = mod_info.amino_acid or "^"
            if mod_info.position == ModPosition.protein_n_term:
                suffix = "[" + suffix
            elif mod_info.position == ModPosition.protein_c_term:
                suffix = "]" + suffix
            elif mod_info.position == ModPosition.n_term:
                suffix = "<" + suffix
            elif mod_info.position == ModPosition.c_term:
                suffix = ">" + suffix
        if not id.endswith(suffix):
            raise ValueError(f"invalid modification id {id}: suffix {suffix} required")
        symbol = id[: -len(suffix)]
        if not symbol.islower() or not symbol.isalnum():
            raise ValueError(
                f"invalid modification id {id}: id must be lowercase alphanum with suffix {suffix}, e.g. mod{suffix}"
            )

    def find_by_amino_acid(self, amino_acid: str) -> List[str]:
        if amino_acid not in self._amino_acid_map:
            return []
        else:
            return self._amino_acid_map[amino_acid]

    def find_by_position(self, position: ModPosition) -> List[str]:
        if position not in self._position_map:
            return []
        else:
            return self._position_map[position]

    def search(self, symbol: str, position: ModPosition = ModPosition.anywhere):
        key = symbol
        if key in self:
            if position == ModPosition.anywhere:
                return key
            else:
                for pos, value in self._position_map.items():
                    if position & pos and key in value:
                        return key
        for pos, suffix in {
            ModPosition.protein_n_term: "[",
            ModPosition.n_term: "<",
            ModPosition.protein_c_term: "]",
            ModPosition.c_term: ">",
        }.items():
            if position & pos:
                key = symbol[:-1] + suffix + symbol[-1]
                if key in self:
                    return key
                key = key[:-1] + "^"
                if key in self:
                    return key
        key = key[:-1] + "*"
        if key in self:
            return key
        return ""

    def search_by_unimod(
        self,
        unimod: int,
        amino_acid: str = "",
        position: ModPosition = ModPosition.anywhere,
    ) -> List[str]:
        if unimod not in self._unimod_map:
            return []
        lr = self._unimod_map[unimod]
        if len(lr) == 0 or amino_acid == "":
            return lr
        la = self.find_by_amino_acid(amino_acid) + self.find_by_amino_acid("")
        lr = [x for x in lr if x in la]
        if len(lr) == 0 or position == ModPosition.anywhere:
            return lr
        lp = []
        for pos, value in self._position_map.items():
            if position & pos:
                lp.extend(value)
        lr = [x for x in lr if x in lp]
        return lr

    def search_by_name(
        self,
        name: str,
        amino_acid: str = "",
        position: ModPosition = ModPosition.anywhere,
    ) -> List[str]:
        lr = []
        exact_name = {key: mod for key, mod in self.items() if mod.name == name}
        start_name = {
            key: mod
            for key, mod in self.items()
            if mod.name.startswith(f"{name} (") and mod.name.endswith(")")
        }

        if amino_acid == "" and position == ModPosition.anywhere:
            lr.extend(exact_name.keys())
            lr.append(start_name.keys())
        elif amino_acid != "":
            for key, mod in exact_name.items():
                if mod.amino_acid == amino_acid and (mod.position & position):
                    lr.append(key)
            for key, mod in exact_name.items():
                if mod.amino_acid == "" and (mod.position & position):
                    lr.append(key)
            for key, mod in start_name.items():
                if mod.amino_acid == amino_acid and (mod.position & position):
                    lr.append(key)
            for key, mod in start_name.items():
                if mod.amino_acid == "" and (mod.position & position):
                    lr.append(key)
        else:
            for key, mod in exact_name.items():
                if mod.position & position:
                    lr.append(key)
            for key, mod in start_name.items():
                if mod.position & position:
                    lr.append(key)
        return lr

    def search_by_mass(
        self,
        mass: float,
        amino_acid: str = "",
        position: ModPosition = ModPosition.anywhere,
        tolerance: float = 0.5,
    ) -> List[str]:
        it = None
        if amino_acid != "":
            it = self.find_by_amino_acid(amino_acid) + self.find_by_amino_acid("")
        if position != ModPosition.anywhere:
            lp = itertools.chain.from_iterable(
                [value for pos, value in self._position_map.items() if position & pos]
            )
            if it is None:
                it = lp
            else:
                it = set(it).intersection(lp)
        if it is None:
            it = self.items()
        else:
            it = ((x, self[x]) for x in it)

        delta = sorted(
            filter(
                lambda t: t[1] <= tolerance,
                ((key, abs(mod.mass - mass)) for key, mod in it),
            ),
            key=lambda t: t[1],
        )

        return [t[0] for t in delta]


AAModPair = Tuple[str, str]
ModifiedSequence = List[AAModPair]


def modified_sequence_to_str(parsed_sequence: ModifiedSequence) -> str:
    s = ""
    for aa, mod in parsed_sequence:
        for ch in mod:
            if ch.isupper() or ch in "[<]>^*":
                break
            s += ch
        s += aa
    return s


class ModifiedSequenceParser:
    def __init__(
        self,
        amino_acids: AminoAcidCollection,
        modifications: ModificationCollection,
    ):
        self.amino_acids = amino_acids
        self.modifications = modifications

    def parse_modified_sequence(self, sequence: str) -> ModifiedSequence:
        parsed = []
        mod_str = ""
        for i, ch in enumerate(sequence):
            mod_str += ch
            if ch.isupper():
                if ch not in self.amino_acids:
                    raise ValueError(f"amino acid not found: {ch}")
                if len(mod_str) > 1:
                    position = ModPosition.not_term
                    if len(parsed) == 0:
                        position |= ModPosition.n_term
                    elif i == len(sequence) - 1:
                        position |= ModPosition.c_term
                    mod = self.modifications.search(
                        mod_str, position=ModPosition(position)
                    )
                    if mod == "":
                        raise ValueError(f"modification not found: {mod_str}")
                    parsed.append((ch, mod))
                else:
                    parsed.append((ch, ""))
                mod_str = ""
        if mod_str != "":
            raise ValueError(f"invalid peptide format: {sequence}")
        return parsed


class ModifiedSequenceFormatterUnimod:
    def __init__(
        self,
        amino_acids: AminoAcidCollection,
        modifications: ModificationCollection,
    ):
        self.amino_acids = amino_acids
        self.modifications = modifications

    def parse_modified_sequence(self, sequence: str) -> ModifiedSequence:
        parsed = []
        terminus = 0
        terminal_mod = ""
        aa = ""
        mod = ""
        t = (None,)
        errorcode = -1
        for t in re.findall(
            r"([_A-Z])(?:\([uU][nN][iI][mM][oO][dD]:([0-9]+)\))?|(.+)", sequence
        ):
            if t[-1]:
                errorcode = -2
                break
            if t[0] == "_":
                terminus += 1
                if terminus == 1 or terminus == 2:
                    terminal_mod = t[1]
                else:
                    errorcode = -3
                    break
            else:
                if terminus == 0:
                    terminus = 1
                elif terminus > 1:
                    errorcode = -3
                    break
                if errorcode:
                    errorcode = 0
                aa = t[0]
                mod = t[1]
            if aa:
                if terminal_mod:
                    if mod:
                        errorcode = -4
                        break
                    mod = terminal_mod
                    terminal_mod = ""
                if aa not in self.amino_acids:
                    errorcode = -5
                    break
                if mod:
                    mod_list = self.modifications.search_by_unimod(
                        unimod=int(mod),
                        amino_acid=aa,
                        position=(
                            ModPosition.n_term
                            if len(parsed) == 0
                            else ModPosition.c_term
                            if terminus == 2
                            else ModPosition.not_term
                        ),
                    )
                    if len(mod_list) == 0:
                        errorcode = -6
                        break
                    mod_ = mod_list[0]
                else:
                    mod_ = ""
                if terminus == 2:
                    parsed.pop()
                parsed.append((aa, mod_))

        if errorcode == -1:
            raise ValueError(
                f"invalid Unimod peptide format {sequence}: no amino acids"
            )
        elif errorcode == -2:
            raise ValueError(
                f"invalid Unimod peptide format {sequence}: unknown pattern {t[-1]}"
            )
        elif errorcode == -3:
            raise ValueError(
                f"invalid Unimod peptide format {sequence}: incorrect terminus symbols"
            )
        elif errorcode == -4:
            raise ValueError(
                f"invalid Unimod peptide{sequence}: terminal and internal modifications ({terminal_mod}, {mod}) at the same AA position are not supported"
            )
        elif errorcode == -5:
            raise ValueError(
                f"invalid Unimod peptide {sequence}: unknown amino acid {aa}"
            )
        elif errorcode == -6:
            raise ValueError(
                f"invalid Unimod peptide {sequence}: unknown modification (Unimod:{mod}) at {aa}"
            )
        elif errorcode:
            raise ValueError(f"invalid Unimod peptide format {sequence}")
        return parsed

    def modified_sequence_to_str(self, parsed_sequence: ModifiedSequence) -> str:
        s = ""
        for i, (aa, mod) in enumerate(parsed_sequence):
            if mod:
                modinfo = self.modifications.get(mod, None)
                if modinfo is None:
                    raise ValueError(f"unknown modification {mod} at {aa}")
                if modinfo.position == ModPosition.anywhere:
                    s += f"{aa}(UniMod:{modinfo.unimod})"
                elif i == 0 and modinfo.position & ModPosition.any_n_term:
                    s += f"_(UniMod:{modinfo.unimod}){aa}"
                elif (
                    i == len(parsed_sequence) - 1
                    and modinfo.position & ModPosition.any_c_term
                ):
                    s += f"{aa}_(UniMod:{modinfo.unimod})"
                else:
                    s += f"{aa}(UniMod:{modinfo.unimod})"
            else:
                s += aa
        if len(s) == 0:
            raise ValueError("empty modified peptide sequence")
        return s


class ModifiedSequenceFormatterTPP:
    def __init__(
        self,
        elements: ElementCollection,
        amino_acids: AminoAcidCollection,
        modifications: ModificationCollection,
    ):
        self.elements = elements
        self.amino_acids = amino_acids
        self.modifications = modifications

    def parse_modified_sequence(self, sequence: str) -> ModifiedSequence:
        parsed = []
        terminus = 0
        terminal_mod: str = ""
        aa: str = ""
        mod: str = ""
        t = (None,)
        errorcode = -1
        for t in re.findall(
            r"([A-Za-z])(?:\[([+\-]?[0-9]*(?:\.[0-9]*)?)\])?|(.+)", sequence
        ):
            if t[-1]:
                errorcode = -2
                break
            if t[0] == "n":
                if terminus != 0:
                    errorcode = -3
                    break
                terminus += 1
                terminal_mod = t[1]
            elif t[0] == "c":
                if terminus != 1:
                    errorcode = -3
                    break
                terminus += 1
                terminal_mod = t[1]
            else:
                if terminus == 0:
                    terminus = 1
                elif terminus > 1:
                    errorcode = -3
                    break
                if errorcode:
                    errorcode = 0
                aa = t[0]
                mod = t[1]
            if aa:
                if terminal_mod and mod:
                    errorcode = -4
                    break
                if aa not in self.amino_acids:
                    errorcode = -5
                    break

                mass = None
                digits = 0
                if mod:
                    mass = float(mod)
                    digits = len(mod) - max(mod.find("."), len(mod) - 1) - 1
                    if mod.startswith("+") or mod.startswith("-"):
                        pass
                    else:
                        mass -= self.amino_acids[aa].mass
                if terminal_mod:
                    mass = float(terminal_mod)
                    digits = (
                        len(terminal_mod)
                        - max(terminal_mod.find("."), len(terminal_mod) - 1)
                        - 1
                    )
                    if terminal_mod.startswith("+") or terminal_mod.startswith("-"):
                        pass
                    elif terminus == 1:
                        mass -= self.elements["H"].mass
                    else:
                        mass -= self.elements["H"].mass + self.elements["O"].mass
                    mod = terminal_mod
                    terminal_mod = ""
                if mass:
                    mod_list = self.modifications.search_by_mass(
                        mass=mass,
                        amino_acid=aa,
                        position=(
                            ModPosition.n_term
                            if len(parsed) == 0
                            else ModPosition.c_term
                            if terminus == 2
                            else ModPosition.not_term
                        ),
                        tolerance=0.1**digits * 0.5,
                    )
                    if len(mod_list) == 0:
                        errorcode = -6
                        break
                    mod_ = mod_list[0]
                else:
                    mod_ = ""
                if terminus == 2:
                    parsed.pop()
                parsed.append((aa, mod_))

        if errorcode == -1:
            raise ValueError(f"invalid TPP peptide format {sequence}: no amino acids")
        elif errorcode == -2:
            raise ValueError(
                f"invalid TPP peptide format {sequence}: unknown pattern {t[-1]}"
            )
        elif errorcode == -3:
            raise ValueError(
                f"invalid TPP peptide format {sequence}: incorrect terminus symbols"
            )
        elif errorcode == -4:
            raise ValueError(
                f"invalid TPP peptide{sequence}: terminal and internal modifications ({terminal_mod}, {mod}) at the same AA position are not supported"
            )
        elif errorcode == -5:
            raise ValueError(f"invalid TPP peptide {sequence}: unknown amino acid {aa}")
        elif errorcode == -6:
            raise ValueError(
                f"invalid TPP peptide {sequence}: unknown modification {mod} at {aa}"
            )
        elif errorcode:
            raise ValueError(f"invalid TPP peptide format {sequence}")
        return parsed

    def modified_sequence_to_str(
        self,
        parsed_sequence: ModifiedSequence,
        num_digits: int = 0,
    ) -> str:
        s = ""
        for i, (aa, mod) in enumerate(parsed_sequence):
            aainfo = self.amino_acids.get(aa, None)
            if aainfo is None:
                raise ValueError(f"unknown amino acid {aa}")
            if mod:
                modinfo = self.modifications.get(mod, None)
                if modinfo is None:
                    raise ValueError(f"unknown modification {mod} at {aa}")
                mass = modinfo.mass
                if modinfo.position == ModPosition.anywhere:
                    mass += aainfo.mass
                    s += f"{aa}[{f'%.{num_digits}f' % mass}]"
                elif i == 0 and modinfo.position & ModPosition.any_n_term:
                    mass += self.elements["H"].mass
                    s += f"n[{f'%.{num_digits}f' % mass}]{aa}"
                elif (
                    i == len(parsed_sequence) - 1
                    and modinfo.position & ModPosition.any_c_term
                ):
                    mass += self.elements["H"].mass + self.elements["O"].mass
                    s += f"{aa}c[{f'%.{num_digits}f' % mass}]"
                else:
                    mass += aainfo.mass
                    s += f"{aa}[{f'%.{num_digits}f' % mass}]"
            else:
                s += aa
        if len(s) == 0:
            raise ValueError("empty modified peptide sequence")
        return s
