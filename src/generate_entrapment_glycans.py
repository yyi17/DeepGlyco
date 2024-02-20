from typing import Union
import pandas as pd

from deepglyco.chem.gpep.glycans import GlycanNode
from deepglyco.chem.gpep.nglycans import NGlycan


glycans = pd.read_csv(r"D:\GlycoDIA\data\background_glycan.txt", header=None)
glycans.columns = ["original_struct"]


def generate_entrapment_glycan(glycan: str):
    new_glycan = NGlycan(GlycanNode.from_str(str(glycan)))
    changed = False
    for branches in new_glycan.branches:
        for branch in branches:
            for node in branch.iter_depth_first():
                if node.monosaccharide == "H":
                    node.monosaccharide = "N"
                    changed = True
    if changed:
        return str(new_glycan.glycan)
    else:
        return None


glycans["entrapment_struct"] = glycans["original_struct"].map(
    generate_entrapment_glycan
)

glycans["original_composition"] = glycans["original_struct"].map(
    lambda x: GlycanNode.from_str(x).composition() if x is not None else None
)
glycans["entrapment_composition"] = glycans["entrapment_struct"].map(
    lambda x: GlycanNode.from_str(x).composition() if x is not None else None
)

glycans["entrapment"] = (
    ~glycans["entrapment_composition"].isin(glycans["original_composition"])
    & ~glycans["entrapment_composition"].isnull()
    & glycans["entrapment_composition"].map(lambda x: x is not None and x["N"] >= 8)
)


glycans.to_csv(r"D:\GlycoDIA\data\entrapment_glycan.csv", index=False)
