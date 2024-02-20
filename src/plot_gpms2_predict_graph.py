# %%
import torch

from typing import List, Literal, Optional
from torch import nn


class IntermediateFeatureExtractor(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        layers: Optional[List[str]] = None,
        target: Literal["input", "output", "both"] = "output",
    ):
        super().__init__()
        self.model = model
        if layers is not None:
            self.layers = layers
        else:
            self.layers = []
        self.target = target

        def update_feature(name, feature):
            assert feature is not None and not isinstance(feature, list)
            f = self.features.get(name, None)
            if isinstance(f, list):
                f.append(feature)
            elif f is not None:
                self.features[name] = [f, feature]
            else:
                self.features[name] = feature

        def build_hook_fn(name):
            if target == "input":

                def hook_fn(module, input, output):
                    update_feature(name, input)

            elif target == "output":

                def hook_fn(module, input, output):
                    update_feature(name, output)

            else:

                def hook_fn(module, input, output):
                    update_feature(name, (input, output))

            return hook_fn


        self.features = {}
        for name, module in self.model.named_modules():
            if layers is None:
                self.layers.append(name)
            elif name not in self.layers:
                continue
            module.register_forward_hook(build_hook_fn(name))

    def forward(self, *args, **kwargs):
        self.features = {}
        output = self.model(*args, **kwargs)
        return output


import os
import pandas as pd

from deepglyco.chem.common.elements import ElementCollection
from deepglyco.chem.gpep.fragments import GlycanFragmentTypeCollection
from deepglyco.chem.gpep.glycans import MonosaccharideCollection
from deepglyco.chem.gpep.gpmass import GlycoPeptideMassCalculator
from deepglyco.chem.pep.aminoacids import AminoAcidCollection
from deepglyco.chem.pep.fragments import PeptideFragmentTypeCollection
from deepglyco.chem.pep.losses import NeutralLossTypeCollection
from deepglyco.chem.pep.mods import ModifiedSequenceParser, ModificationCollection
from deepglyco.deeplib.gpep.ms2b.data import GlycoPeptideBranchMS2OutputConverter
from deepglyco.deeplib.gpep.ms2b.prediction import GlycoPeptideBranchMS2Predictor
from deepglyco.util.di import Context
from deepglyco.util.log import get_logger
from deepglyco.util.progress import TqdmProgressFactory


def register_dependencies(**kwargs):
    ctx = Context()
    ctx.register(
        "elements",
        ElementCollection.load,
        element_file=kwargs.get("element_file", None),
    )
    ctx.register(
        "amino_acids",
        AminoAcidCollection.load,
        amino_acid_file=kwargs.get("amino_acid_file", None),
    )
    ctx.register(
        "modifications",
        ModificationCollection.load,
        modification_file=kwargs.get("modification_file", None),
    )
    ctx.register(
        "monosaccharides",
        MonosaccharideCollection.load,
        monosaccharide_file=kwargs.get("monosaccharide_file", None),
    )
    ctx.register(
        "neutral_loss_types",
        NeutralLossTypeCollection.load,
        neutral_loss_file=kwargs.get("neutral_loss_file", None),
    )
    ctx.register(
        "peptide_fragment_types",
        PeptideFragmentTypeCollection.load,
        peptide_fragment_file=kwargs.get("peptide_fragment_file", None),
    )
    ctx.register(
        "glycan_fragment_types",
        GlycanFragmentTypeCollection.load,
        glycan_fragment_file=kwargs.get("glycan_fragment_file", None),
    )
    ctx.register("sequence_parser", ModifiedSequenceParser)
    ctx.register("mass_calculator", GlycoPeptideMassCalculator)

    ctx.register("converter", GlycoPeptideBranchMS2OutputConverter)

    ctx.register("progress_factory", TqdmProgressFactory)
    ctx.register(
        "logger",
        get_logger,
        name=kwargs.get("log_name", None),
        file=kwargs.get("log_file", None),
    )
    return ctx


# %%
import numpy as np
from deepglyco.chem.gpep.glycans import GlycanNode
from deepglyco.chem.gpep.glycans import GlycanNodeGraph
from deepglyco.chem.gpep.fragments import GlycanFragmentGraph
from deepglyco.chem.gpep.fragments import reducing_end_fragment_graph
from deepglyco.chem.gpep.fragments import branch_fragment_graph


def get_fragment_graph(glycan_struct, features, branch_fragments=False):
    if branch_fragments:
        fragment_graph_fn = branch_fragment_graph
        feat_fragment = "gly_output.output_branch"
        feat_composition = "gly_output", 1
        feat_frag_attn = "gly_output.fragment_branch.attention_sum.gate_nn"
    else:
        fragment_graph_fn = reducing_end_fragment_graph
        feat_fragment = "gly_output.output"
        feat_composition = "gly_output", 0
        feat_frag_attn = "gly_output.fragment.attention_sum.gate_nn"


    fragment_graph = fragment_graph_fn(GlycanNode.from_str(glycan_struct))

    node_values = {
        "fragment": features[feat_fragment].detach().cpu().numpy().sum(axis=1),
        "composition": features[feat_composition[0]][feat_composition[1]].detach().cpu().numpy().sum(axis=1)
    }

    num_cleavages = list(map(len, fragment_graph.fragment_nodes))
    num_cleavages = np.array([
        num_cleavages[f]
        for c, f in fragment_graph.cleavage_fragment_edges
    ])
    cleavage_fragment_edge_values = np.zeros(len(num_cleavages))
    for attn in features[feat_frag_attn]:
        cleavage_fragment_edge_values[num_cleavages == attn.size(1)] = \
        torch.nn.functional.softmax(attn, dim=1).detach().cpu().numpy().flatten()

    edge_values = {
        "combine": np.array([
            node_values["fragment"][f] / node_values["composition"][c]
            if node_values["composition"][c] != 0 else 0
            for f, c in fragment_graph.fragment_composition_edges
        ]),
        "join": cleavage_fragment_edge_values
    }

    return fragment_graph, node_values, edge_values


# %%
def int_to_roman(num: int):
    if num == 0:
        return '0'

    number = [1, 4, 5, 9, 10, 40, 50, 90, 100, 400, 500, 900, 1000]
    symbol = ['I', 'IV', 'V', 'IX', 'X', 'XL', 'L', 'XC', 'C', 'CD', 'D', 'CM', 'M']

    res = []
    i = len(number) - 1
    while num:
        div = num // number[i]
        num %= number[i]
        while div:
            res.append(symbol[i])
            div -= 1
        i -= 1

    return ''.join(res)

# %%
import networkx as nx
import matplotlib.pyplot as plt
from typing import Union
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm


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


# %%
def plot_node_graph(node_graph: Union[GlycanNodeGraph, GlycanFragmentGraph], ax=None):
    monosaccharide_colors = {
        "H": "#00A651",
        "N": "#0072BC",
        "A": "#A54399",
        "G": "#8FCCE9",
        "F": "#ED1C24",
    }
    monosaccharide_shapes = {
        "H": "o",
        "N": "s",
        "A": "d",
        "G": "d",
        "F": "v",
    }

    G = nx.Graph()
    for i, g in enumerate(
        node_graph.nodes
        if isinstance(node_graph, GlycanNodeGraph)
        else node_graph.monosaccharide_nodes
    ):
        G.add_node(i + 1, category="monosaccharide", monosaccharide=g.monosaccharide)
    for x, y in (
        node_graph.edges
        if isinstance(node_graph, GlycanNodeGraph)
        else node_graph.monosaccharide_edges
    ):
        G.add_edge(x + 1, y + 1, label=int_to_roman(y + 1), relation="link")

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
    # nx.drawing.nx_pylab.draw_networkx_edge_labels(
    #     G,
    #     pos=pos,
    #     edge_labels=nx.get_edge_attributes(G, 'label'),
    #     ax=ax,
    #     font_size=6,
    # )

    return ax

# %%
def plot_fragment_graph(
    fragment_graph: GlycanFragmentGraph,
    node_values: dict,
    edge_values: dict,
    right_to_left=False,
    branch_fragments=False,
    node_value_cutoff=0.0,
    ax=None,
):
    G = nx.MultiGraph()

    # monosaccharide_nodes = []
    # for i, g in enumerate(fragment_graph.monosaccharide_nodes):
    #     node_id = f"{g.monosaccharide}{i + 1}"
    #     monosaccharide_nodes.append(node_id)
    #     G.add_node(
    #         node_id,
    #         category_index=-3 if right_to_left else 3,
    #         category="monosaccharide",
    #         monosaccharide=g.monosaccharide,
    #     )
    # for x, y in fragment_graph.monosaccharide_edges:
    #     G.add_edge(monosaccharide_nodes[x], monosaccharide_nodes[y], relation="link")

    cleavage_nodes = []
    for i, g in enumerate(fragment_graph.cleavage_nodes):
        node_id = int_to_roman(g + 1)
        cleavage_nodes.append(node_id)
        G.add_node(
            node_id,
            category_index=-2 if right_to_left else 2,
            category="cleavage",
        )
    # for x, y in fragment_graph.lost_monosaccharide_cleavage_edges:
    #     G.add_edge(monosaccharide_nodes[x], cleavage_nodes[y], relation="lost")
    # for x, y in fragment_graph.retained_monosaccharide_cleavage_edges:
    #     G.add_edge(monosaccharide_nodes[x], cleavage_nodes[y], relation="retained")

    fragment_nodes = []
    for i, g in enumerate(fragment_graph.fragment_nodes):
        node_id = ' ' + ','.join((int_to_roman(gg + 1) for gg in g)) + ' '
        fragment_nodes.append(node_id)
        value = node_values["fragment"][i]
        if node_value_cutoff > 0 and value <= node_value_cutoff:
            continue
        G.add_node(
            node_id,
            category_index=-1 if right_to_left else 1,
            category="fragment",
            value=value,
        )
    for i, (x, y) in enumerate(fragment_graph.cleavage_fragment_edges):
        if node_value_cutoff > 0 and node_values["fragment"][y] <= node_value_cutoff:
            continue
        G.add_edge(
            cleavage_nodes[x],
            fragment_nodes[y],
            relation="join",
            value=edge_values["join"][i],
        )

    composition_nodes = []
    for i, g in enumerate(fragment_graph.composition_nodes):
        node_id = "".join([f"{k}({v})" for k, v in g.items()])
        if not branch_fragments:
            node_id = "   Y-" + (node_id or "0")
        composition_nodes.append(node_id)
        value = node_values["composition"][i]
        if node_value_cutoff > 0 and value <= node_value_cutoff:
            continue
        G.add_node(
            node_id,
            category_index=0,
            category="composition",
            value=value
        )
    for i, (x, y) in enumerate(fragment_graph.fragment_composition_edges):
        if node_value_cutoff > 0 and (
            node_values["composition"][y] <= node_value_cutoff or
            node_values["fragment"][x] <= node_value_cutoff
        ):
            continue
        G.add_edge(
            fragment_nodes[x],
            composition_nodes[y],
            relation="combine",
            value=edge_values["combine"][i]
        )

    pos = multipartite_layout(G, subset_key="category_index", scale=-1)

    if ax is None:
        ax = plt.axes()

    if branch_fragments:
        gly_cmap = LinearSegmentedColormap.from_list("gly", ['lightgrey', '#ff5a76', '#ff3355', '#950019'])
    else:
        gly_cmap = LinearSegmentedColormap.from_list("gly", ['lightgrey', '#d5a1db', '#c277ca', '#6f2e76'])
    comb_cmap = LinearSegmentedColormap.from_list("comb", ['lightgrey', '#339dff', '#194e7f'])

    # for monosaccharide in set(
    #     [
    #         d["monosaccharide"]
    #         for x, d in G.nodes(data=True)
    #         if d["category"] == "monosaccharide"
    #     ]
    # ):
    #     nx.drawing.nx_pylab.draw_networkx_nodes(
    #         G,
    #         pos=pos,
    #         nodelist=[
    #             x
    #             for x, d in G.nodes(data=True)
    #             if d["category"] == "monosaccharide"
    #             and d["monosaccharide"] == monosaccharide
    #         ],
    #         node_color=monosaccharide_colors[monosaccharide],
    #         node_shape=monosaccharide_shapes[monosaccharide],
    #         node_size=100,
    #         ax=ax,
    #     )

    nx.drawing.nx_pylab.draw_networkx_nodes(
        G,
        pos=pos,
        nodelist=[x for x, d in G.nodes(data=True) if d["category"] == "cleavage"],
        node_color="#ffd247",
        node_shape="o",
        node_size=50,
        ax=ax,
    )
    nodelist = [(x, d["value"]) for x, d in G.nodes(data=True) if d["category"] == "fragment"]
    nx.drawing.nx_pylab.draw_networkx_nodes(
        G,
        pos=pos,
        nodelist=[x for x, d in nodelist],
        node_color=np.log1p([v for x, v in nodelist]), # type: ignore
        cmap=gly_cmap,
        vmin=0.0,
        node_size=50,
        ax=ax,
    )
    nodelist = [(x, d["value"]) for x, d in G.nodes(data=True) if d["category"] == "composition"]
    nx.drawing.nx_pylab.draw_networkx_nodes(
        G,
        pos=pos,
        nodelist=[x for x, d in nodelist],
        node_color=np.log1p([v for x, v in nodelist]), # type: ignore
        cmap=gly_cmap,
        vmin=0.0,
        node_shape="o",
        node_size=50,
        margins=(1, 0),
        ax=ax,
    )

    # nx.drawing.nx_pylab.draw_networkx_edges(
    #     G,
    #     pos=pos,
    #     edgelist=[(x, y) for x, y, d in G.edges(data=True) if d["relation"] == "lost"],
    #     edge_color="red",
    #     ax=ax,
    #     width=0.5,
    # )
    # nx.drawing.nx_pylab.draw_networkx_edges(
    #     G,
    #     pos=pos,
    #     edgelist=[
    #         (x, y) for x, y, d in G.edges(data=True) if d["relation"] == "retained"
    #     ],
    #     edge_color="green",
    #     ax=ax,
    #     width=0.5,
    # )
    edgelist=[(x, y, d["value"]) for x, y, d in G.edges(data=True) if d["relation"] == "join"]
    nx.drawing.nx_pylab.draw_networkx_edges(
        G,
        pos=pos,
        edgelist=[(x, y) for x, y, v in edgelist],
        edge_color=[v for x, y, v in edgelist], # type: ignore
        edge_cmap=comb_cmap,
        edge_vmin=0.0,
        width=np.array([v for x, y, v in edgelist]) + 0.5, # type: ignore
        ax=ax,
    )
    edgelist=[(x, y, d["value"]) for x, y, d in G.edges(data=True) if d["relation"] == "combine"]
    nx.drawing.nx_pylab.draw_networkx_edges(
        G,
        pos=pos,
        edgelist=[(x, y) for x, y, v in edgelist],
        edge_color=[v for x, y, v in edgelist], # type: ignore
        edge_cmap=comb_cmap,
        edge_vmin=0.0,
        width=np.array([v for x, y, v in edgelist]) + 0.5, # type: ignore
        ax=ax,
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
        horizontalalignment="right" if right_to_left else "left",
    )

    return ax




# %%
pretrained_file = r"gpms2b_model.pt"

ctx = register_dependencies()

predictor = ctx.build(GlycoPeptideBranchMS2Predictor)
predictor.load_model(pretrained_file)

logger = ctx.get("logger")
if logger:
    logger.info(f"Use configs: {predictor.get_configs()}")

predictor.model = IntermediateFeatureExtractor(predictor.model)


# %%
sequence = "NPNGTVTVISR"
glycan_position = 3
glycan_struct = '(N(N(H(H(N(H)))(H(N(H)))))(F))'
precursor_charge = 3

out_file = f"{sequence}_{glycan_position}_{glycan_struct}_{precursor_charge}.svg"

batch = predictor._collate_data([
    predictor.converter.glycopeptide_to_tensor(
        sequence=sequence,
        glycan_struct=glycan_struct,
        glycan_position=glycan_position,
        precursor_charge=precursor_charge,
    )
])
pred = predictor.model(batch.to(predictor.device)).to("cpu")

features = predictor.model.features


# %%
fragment_graph, node_values, edge_values = get_fragment_graph(
    glycan_struct,
    features,
    branch_fragments=False,
)

ax = plot_fragment_graph(
    fragment_graph,
    node_values=node_values,
    edge_values=edge_values,
    branch_fragments=False,
    right_to_left=False,
    node_value_cutoff=0.01,
)
ax.axis('off')

ax.figure.set_size_inches(12 / 2.54, 25 / 2.54)
ax.figure.savefig(
    os.path.splitext(out_file)[0] + ".fraggraph.svg",
    transparent=True,
    bbox_inches='tight',
)

plt.show(ax.figure)
plt.close(ax.figure)

# cb = plt.colorbar(cm.ScalarMappable(cmap=ax.collections[1].cmap), cax=plt.subplot())
# cb.ax.figure.set_size_inches(0.25 / 2.54, 8 / 2.54)
# cb.ax.figure.savefig(
#     f"purple.colorbar.svg",
#     transparent=True,
#     bbox_inches='tight',
# )

# cb = plt.colorbar(cm.ScalarMappable(cmap=ax.collections[3].cmap), cax=plt.subplot())
# cb.ax.figure.set_size_inches(0.25 / 2.54, 8 / 2.54)
# cb.ax.figure.savefig(
#     f"blue.colorbar.svg",
#     transparent=True,
#     bbox_inches='tight',
# )

# %
fragment_graph, node_values, edge_values = get_fragment_graph(
    glycan_struct,
    features,
    branch_fragments=True,
)

ax = plot_fragment_graph(
    fragment_graph,
    node_values=node_values,
    edge_values=edge_values,
    branch_fragments=True,
    right_to_left=True,
    node_value_cutoff=0.01,
)
ax.axis('off')

ax.figure.set_size_inches(12 / 2.54, 15 / 2.54)
ax.figure.savefig(
    os.path.splitext(out_file)[0] + ".bfraggraph.svg",
    transparent=True,
    bbox_inches='tight',
)

plt.show(ax.figure)
plt.close(ax.figure)

# cb = plt.colorbar(cm.ScalarMappable(cmap=ax.collections[1].cmap), cax=plt.subplot())
# cb.ax.figure.set_size_inches(0.25 / 2.54, 8 / 2.54)
# cb.ax.figure.savefig(
#     f"plots/gpscore/red.colorbar.svg",
#     transparent=True,
#     bbox_inches='tight',
# )

# %
ax = plot_node_graph(
    fragment_graph,
)
ax.axis('off')

ax.figure.set_size_inches(8 / 2.54, 4 / 2.54)
ax.figure.savefig(
    os.path.splitext(out_file)[0] + ".nodegraph.svg",
    transparent=True,
    bbox_inches='tight',
)

plt.show(ax.figure)
plt.close(ax.figure)

