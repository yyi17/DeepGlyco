# %%
import os

os.chdir("..")
print(os.getcwd())


# %%
from deepglyco.chem.common.elements import ElementCollection

elements = ElementCollection.load()
print(elements)

print(elements.parse_element_composition("C H(2) O"))
# {'C': 1, 'H': 2, 'O': 1}
print(elements.parse_element_composition("H(-1) N(-1) O"))
# {'H': -1, 'N': -1, 'O': 1}

print(elements.mass_from_element_composition("C H(2) O"))
# 30.0105647
print(elements.mass_from_element_composition("H(-1) N(-1) O"))
# 0.9840155950000007


# %%
from deepglyco.chem.pep.aminoacids import AminoAcidCollection

amino_acids = AminoAcidCollection.load(elements)
print(amino_acids)


# %%
from deepglyco.chem.pep.losses import NeutralLossTypeCollection

neutral_loss_types = NeutralLossTypeCollection.load(elements)
print(neutral_loss_types)


# %%
from deepglyco.chem.pep.mods import ModificationCollection

modifications = ModificationCollection.load(elements)
print(modifications)

from deepglyco.chem.pep.mods import ModifiedSequenceParser

seq_parser = ModifiedSequenceParser(amino_acids, modifications)
print(seq_parser.parse_modified_sequence("AANDAGYFNDEoxMAPIEVacKTK"))

# %%
from deepglyco.chem.pep.mods import ModifiedSequenceFormatterUnimod

seq_parser = ModifiedSequenceFormatterUnimod(amino_acids, modifications)

for sequence in [
    "_AANDAGYFNDEM(unimod:35)APIEVK(unimod:1)TK_",
    "_(UniMod:1)PEPC(UniMod:4)PEPM(UniMod:35)PEPR_(UniMod:2)",
]:
    parsed_sequence = seq_parser.parse_modified_sequence(sequence)
    print(parsed_sequence)
    parsed_sequence = seq_parser.modified_sequence_to_str(parsed_sequence)
    print(parsed_sequence, sequence)

# %%
from deepglyco.chem.pep.mods import ModifiedSequenceFormatterTPP

seq_parser = ModifiedSequenceFormatterTPP(elements, amino_acids, modifications)

for sequence in [
    "AANDAGYFNDEM[147]APIEVK[170]TK",
    "n[43]PEPC[160]PEPM[147]PEPRc[16]",
]:
    parsed_sequence = seq_parser.parse_modified_sequence(sequence)
    print(parsed_sequence)
    parsed_sequence = seq_parser.modified_sequence_to_str(parsed_sequence)
    print(parsed_sequence, sequence)

for sequence in [
    "AANDAGYFNDEM[+15.99]APIEVK[+42.01]TK",
    "n[+42]PEPC[+57]PEPM[+16]PEPRc[-1]",
]:
    parsed_sequence = seq_parser.parse_modified_sequence(sequence)
    print(parsed_sequence)
    parsed_sequence = seq_parser.modified_sequence_to_str(parsed_sequence, num_digits=2)
    print(parsed_sequence, sequence)


# %%
from deepglyco.chem.common.elements import ElementCollection
from deepglyco.chem.pep.aminoacids import AminoAcidCollection
from deepglyco.chem.pep.fragments import PeptideFragmentTypeCollection
from deepglyco.chem.pep.losses import NeutralLossTypeCollection
from deepglyco.chem.pep.mods import ModifiedSequenceParser, ModificationCollection
from deepglyco.chem.pep.pepmass import PeptideMassCalculator
from deepglyco.util.di import Context

ctx = Context()
ctx.register("elements", ElementCollection.load)
ctx.register("amino_acids", AminoAcidCollection.load)
ctx.register("modifications", ModificationCollection.load)
ctx.register("neutral_loss_types", NeutralLossTypeCollection.load)
ctx.register("peptide_fragment_types", PeptideFragmentTypeCollection.load)

seq_parser = ctx.build(ModifiedSequenceParser)
mass_calc = ctx.build(PeptideMassCalculator)

sequence = seq_parser.parse_modified_sequence("AANDAGYFNDEoxMAPIEVacKTK")

print(mass_calc.peptide_mass(sequence))
# 2241.0304413899994
print(mass_calc.precursor_mz(sequence, charge=1))
# 2242.037717856869
print(mass_calc.precursor_mz(sequence))
# [1121.52249716  748.0174236 ]

print(mass_calc.fragment_mass(sequence))
print(
    mass_calc.fragment_mass(sequence, fragment_type="b", loss_type=["", "NH3", "H2O"])
)

print(mass_calc.fragment_mass(sequence, return_annotation=False))
print(
    mass_calc.fragment_mass(
        sequence,
        fragment_type="b",
        loss_type=["", "NH3", "H2O"],
        return_annotation=False,
    )
)

print(mass_calc.fragment_mz(sequence))
print(mass_calc.fragment_mz(sequence, loss_type=["", "NH3", "H2O"]))

print(mass_calc.fragment_mz(sequence, return_annotation=False))
print(
    mass_calc.fragment_mz(
        sequence,
        loss_type=["", "NH3", "H2O"],
        return_annotation=False,
    )
)

# %%
from deepglyco.chem.common.elements import ElementCollection
from deepglyco.chem.gpep.fragments import GlycanFragmentTypeCollection
from deepglyco.chem.gpep.glycans import GlycanNode, MonosaccharideCollection
from deepglyco.chem.gpep.gpmass import GlycoPeptideMassCalculator
from deepglyco.chem.pep.aminoacids import AminoAcidCollection
from deepglyco.chem.pep.fragments import PeptideFragmentTypeCollection
from deepglyco.chem.pep.losses import NeutralLossTypeCollection
from deepglyco.chem.pep.mods import ModifiedSequenceParser, ModificationCollection
from deepglyco.util.di import Context

ctx = Context()
ctx.register("elements", ElementCollection.load)
ctx.register("amino_acids", AminoAcidCollection.load)
ctx.register("modifications", ModificationCollection.load)
ctx.register("monosaccharides", MonosaccharideCollection.load)
ctx.register("neutral_loss_types", NeutralLossTypeCollection.load)
ctx.register("peptide_fragment_types", PeptideFragmentTypeCollection.load)
ctx.register("glycan_fragment_types", GlycanFragmentTypeCollection.load)

seq_parser = ctx.build(ModifiedSequenceParser)
mass_calc = ctx.build(GlycoPeptideMassCalculator)

sequence = seq_parser.parse_modified_sequence("LRPIVIAMNYSLPLR")
glycan = GlycanNode.from_str("(N(N(H(H(H(H(H)))(H))(H(H)(H)))))")
glycan_position = 9

print(mass_calc.glycopeptide_mass(sequence, glycan))
print(mass_calc.precursor_mz(sequence, glycan))

print(
    mass_calc.peptide_fragment_mass(
        sequence, glycan_position=glycan_position, fragment_glycan=["$", "N(1)"]
    )
)

print(mass_calc.glycan_fragment_mass(sequence, glycan=glycan))
print(
    mass_calc.fragment_mass(
        sequence,
        glycan=glycan,
        glycan_position=glycan_position,
    )
)
print(
    mass_calc.fragment_mz(
        sequence,
        glycan=glycan,
        glycan_position=glycan_position,
        glycan_fragment_charge=[1, 2, 3],
    )
)

# %%
from deepglyco.chem.gpep.glycans import GlycanNode
from deepglyco.chem.gpep.fragments import reducing_end_fragments, branch_fragments

glycan = GlycanNode.from_str("(N(N(H(H(N(H(A(A)))(A)))(H(N(H)(F))))))")

print(reducing_end_fragments(glycan))

print(branch_fragments(glycan))

# %%
print(mass_calc.glycan_fragment_mass(sequence, glycan=glycan, fragment_type=["Y", "B"]))

print(
    mass_calc.fragment_mz(
        sequence,
        glycan=glycan,
        glycan_position=glycan_position,
        glycan_fragment_type=["Y", "B"],
    )
)

# %%
from deepglyco.chem.gpep.fragments import reducing_end_fragment_graph

print(reducing_end_fragment_graph(glycan).composition_nodes)
print(
    reducing_end_fragments(glycan)
    == reducing_end_fragment_graph(glycan).composition_nodes[1:]
)


# %%
from deepglyco.chem.gpep.fragments import branch_fragment_graph

print(branch_fragment_graph(glycan).composition_nodes)
print(branch_fragments(glycan) == branch_fragment_graph(glycan).composition_nodes)


# %%
import networkx as nx
import matplotlib.pyplot as plt
from typing import Union

from deepglyco.chem.gpep.glycans import GlycanNodeGraph
from deepglyco.chem.gpep.fragments import GlycanFragmentGraph


def multipartite_layout(G, subset_key="subset", align="vertical", **kwargs):
    import numpy as np

    pos = nx.layout.multipartite_layout(
        G,
        subset_key=subset_key,
        align=align,
        **kwargs,
    )

    layers = {}
    for v, data in G.nodes(data=True):
        layer = data[subset_key]
        layers[layer] = [v] + layers.get(layer, [])

    if align == "horizontal":
        index = 0
    else:
        index = 1

    ranges = {}
    for layer, nodes in layers.items():
        pos_layer = np.array([pos[node][index] for node in nodes])
        ranges[layer] = [pos_layer.min(), pos_layer.max()]

    largest_range = [
        min([r[0] for layer, r in ranges.items()]),
        max([r[1] for layer, r in ranges.items()]),
    ]

    for layer, nodes in layers.items():
        if ranges[layer][1] == ranges[layer][0]:
            continue
        scale = (largest_range[1] - largest_range[0]) / (
            ranges[layer][1] - ranges[layer][0]
        )
        for node in nodes:
            pos[node][index] = (
                pos[node][index] - ranges[layer][0]
            ) * scale + largest_range[0]

    return pos


monosaccharide_colors = {
    "H": "green",
    "N": "blue",
    "A": "purple",
    "G": "lightgray",
    "F": "red",
}

monosaccharide_shapes = {
    "H": "o",
    "N": "s",
    "A": "d",
    "G": "d",
    "F": "v",
}


def plot_node_graph(node_graph: Union[GlycanNodeGraph, GlycanFragmentGraph], ax=None):
    G = nx.Graph()
    for i, g in enumerate(
        node_graph.nodes
        if isinstance(node_graph, GlycanNodeGraph)
        else node_graph.monosaccharide_nodes
    ):
        G.add_node(i, category="monosaccharide", monosaccharide=g.monosaccharide)
    for x, y in (
        node_graph.edges
        if isinstance(node_graph, GlycanNodeGraph)
        else node_graph.monosaccharide_edges
    ):
        G.add_edge(x, y, relation="link")

    pos = nx.layout.kamada_kawai_layout(G, scale=-1)

    if ax is None:
        ax = plt.axes()
    for monosaccharide in set(
        [
            d["monosaccharide"]
            for x, d in G.nodes(data=True)
            if d["category"] == "monosaccharide"
        ]
    ):
        nx.drawing.nx_pylab.draw_networkx_nodes(
            G,
            pos=pos,
            nodelist=[
                x
                for x, d in G.nodes(data=True)
                if d["category"] == "monosaccharide"
                and d["monosaccharide"] == monosaccharide
            ],
            node_color=monosaccharide_colors[monosaccharide],
            node_shape=monosaccharide_shapes[monosaccharide],
            node_size=100,
            ax=ax,
        )

    nx.drawing.nx_pylab.draw_networkx_edges(
        G,
        pos=pos,
        edgelist=[(x, y) for x, y, d in G.edges(data=True) if d["relation"] == "link"],
        ax=ax,
    )
    nx.drawing.nx_pylab.draw_networkx_labels(
        G,
        pos=pos,
        font_size=8,
        ax=ax,
    )

    return ax


def plot_fragment_graph(fragment_graph: GlycanFragmentGraph, ax=None):
    G = nx.MultiGraph()

    monosaccharide_nodes = []
    for i, g in enumerate(fragment_graph.monosaccharide_nodes):
        node_id = f"{g.monosaccharide}{i}"
        monosaccharide_nodes.append(node_id)
        G.add_node(
            node_id,
            category_index=3,
            category="monosaccharide",
            monosaccharide=g.monosaccharide,
        )
    for x, y in fragment_graph.monosaccharide_edges:
        G.add_edge(monosaccharide_nodes[x], monosaccharide_nodes[y], relation="link")

    cleavage_nodes = []
    for i, g in enumerate(fragment_graph.cleavage_nodes):
        node_id = f"c{g}"
        cleavage_nodes.append(node_id)
        G.add_node(node_id, category_index=2, category="cleavage")
    for x, y in fragment_graph.lost_monosaccharide_cleavage_edges:
        G.add_edge(monosaccharide_nodes[x], cleavage_nodes[y], relation="lost")
    for x, y in fragment_graph.retained_monosaccharide_cleavage_edges:
        G.add_edge(monosaccharide_nodes[x], cleavage_nodes[y], relation="retained")

    fragment_nodes = []
    for i, g in enumerate(fragment_graph.fragment_nodes):
        node_id = f"f{g}"
        fragment_nodes.append(node_id)
        G.add_node(node_id, category_index=1, category="fragment")
    for x, y in fragment_graph.cleavage_fragment_edges:
        G.add_edge(cleavage_nodes[x], fragment_nodes[y], relation="join")

    composition_nodes = []
    for i, g in enumerate(fragment_graph.composition_nodes):
        node_id = f"{i}:   " + "".join([f"{k}{v}" for k, v in g.items()])
        composition_nodes.append(node_id)
        G.add_node(node_id, category_index=0, category="composition")
    for x, y in fragment_graph.fragment_composition_edges:
        G.add_edge(fragment_nodes[x], composition_nodes[y], relation="combine")

    pos = multipartite_layout(G, subset_key="category_index", scale=-1)

    if ax is None:
        ax = plt.axes()
    for monosaccharide in set(
        [
            d["monosaccharide"]
            for x, d in G.nodes(data=True)
            if d["category"] == "monosaccharide"
        ]
    ):
        nx.drawing.nx_pylab.draw_networkx_nodes(
            G,
            pos=pos,
            nodelist=[
                x
                for x, d in G.nodes(data=True)
                if d["category"] == "monosaccharide"
                and d["monosaccharide"] == monosaccharide
            ],
            node_color=monosaccharide_colors[monosaccharide],
            node_shape=monosaccharide_shapes[monosaccharide],
            node_size=100,
            ax=ax,
        )

    nx.drawing.nx_pylab.draw_networkx_nodes(
        G,
        pos=pos,
        nodelist=[x for x, d in G.nodes(data=True) if d["category"] == "cleavage"],
        node_color="gold",
        node_shape="o",
        node_size=50,
        ax=ax,
    )
    nx.drawing.nx_pylab.draw_networkx_nodes(
        G,
        pos=pos,
        nodelist=[x for x, d in G.nodes(data=True) if d["category"] == "fragment"],
        node_color="pink",
        node_shape="o",
        node_size=50,
        ax=ax,
    )
    nx.drawing.nx_pylab.draw_networkx_nodes(
        G,
        pos=pos,
        nodelist=[x for x, d in G.nodes(data=True) if d["category"] == "composition"],
        node_color="skyblue",
        node_shape="o",
        node_size=50,
        margins=(0.25, 0),
        ax=ax,
    )

    nx.drawing.nx_pylab.draw_networkx_edges(
        G,
        pos=pos,
        edgelist=[(x, y) for x, y, d in G.edges(data=True) if d["relation"] == "lost"],
        edge_color="red",
        ax=ax,
        width=0.5,
    )
    nx.drawing.nx_pylab.draw_networkx_edges(
        G,
        pos=pos,
        edgelist=[
            (x, y) for x, y, d in G.edges(data=True) if d["relation"] == "retained"
        ],
        edge_color="green",
        ax=ax,
        width=0.5,
    )
    nx.drawing.nx_pylab.draw_networkx_edges(
        G,
        pos=pos,
        edgelist=[(x, y) for x, y, d in G.edges(data=True) if d["relation"] == "join"],
        ax=ax,
        width=0.5,
    )
    nx.drawing.nx_pylab.draw_networkx_edges(
        G,
        pos=pos,
        edgelist=[
            (x, y) for x, y, d in G.edges(data=True) if d["relation"] == "combine"
        ],
        ax=ax,
        width=0.5,
    )

    nx.drawing.nx_pylab.draw_networkx_labels(
        G,
        pos=pos,
        labels={n: n if d["category"] != "composition" else "" for n, d in G.nodes(data=True)},
        font_size=8,
        ax=ax,
        horizontalalignment="center",
    )
    nx.drawing.nx_pylab.draw_networkx_labels(
        G,
        pos=pos,
        labels={n: n if d["category"] == "composition" else "" for n, d in G.nodes(data=True)},
        font_size=8,
        ax=ax,
        horizontalalignment="left",
    )

    return ax


# %%
fragment_graph = branch_fragment_graph(glycan)

plt.figure(figsize=(8, 8))
ax = plt.subplot(1, 4, 1)
plot_node_graph(fragment_graph, ax=ax)
ax = plt.subplot(1, 4, (2, 4))
plot_fragment_graph(fragment_graph, ax=ax)

# %%
fragment_graph = reducing_end_fragment_graph(glycan)

plt.figure(figsize=(20, 20))
ax = plt.subplot(1, 4, 1)
plot_node_graph(fragment_graph, ax=ax)
ax = plt.subplot(1, 4, (2, 4))
plot_fragment_graph(fragment_graph, ax=ax)

# %%
