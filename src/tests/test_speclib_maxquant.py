# %%
import os

os.chdir("..")
print(os.getcwd())

# %%
from deepglyco.chem.common.elements import ElementCollection
from deepglyco.chem.pep.aminoacids import AminoAcidCollection
from deepglyco.chem.pep.mods import ModificationCollection
from deepglyco.speclib.pep.parser import MaxQuantReportParser
from deepglyco.util.di import Context

ctx = Context()
ctx.register("elements", ElementCollection.load)
ctx.register("amino_acids", AminoAcidCollection.load)
ctx.register("modifications", ModificationCollection.load)
mqparser: MaxQuantReportParser = ctx.build(MaxQuantReportParser)
print(mqparser.parse_modified_sequence("_AANDAGYFNDEM(ox)APIEVK(ac)TK_"))

# %%
