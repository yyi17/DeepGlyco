"""
Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks
https://arxiv.org/abs/1503.00075

Modifications
- Add num_layers
"""

__all__ = ["TreeLSTM"]

from typing import cast

import dgl
import torch as th
import torch.nn as nn

# class TreeLSTMCell(nn.Module):
#     def __init__(self, x_size, h_size):
#         super(TreeLSTMCell, self).__init__()
#         self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
#         self.U_iou = nn.Linear(2 * h_size, 3 * h_size, bias=False)
#         self.b_iou = nn.parameter.Parameter(th.zeros(1, 3 * h_size))
#         self.U_f = nn.Linear(2 * h_size, 2 * h_size)

#     def message_func(self, edges):
#         return {"h": edges.src["h"], "c": edges.src["c"]}

#     def reduce_func(self, nodes):
#         h_cat = nodes.mailbox["h"].view(nodes.mailbox["h"].size(0), -1)
#         f = th.sigmoid(self.U_f(h_cat)).view(*nodes.mailbox["h"].size())
#         c = th.sum(f * nodes.mailbox["c"], 1)
#         return {"iou": self.U_iou(h_cat), "c": c}

#     def apply_node_func(self, nodes):
#         iou = nodes.data["iou"] + self.b_iou
#         i, o, u = th.chunk(iou, 3, 1)
#         i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
#         c = i * u + nodes.data["c"]
#         h = o * th.tanh(c)
#         return {"h": h, "c": c}


class ChildSumTreeLSTMCell(nn.Module):
    def __init__(self, x_size, h_size):
        super(ChildSumTreeLSTMCell, self).__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=False)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.b_iou = nn.parameter.Parameter(th.zeros(1, 3 * h_size))
        self.U_f = nn.Linear(h_size, h_size)

    def message_func(self, edges):
        return {"h": edges.src["h"], "c": edges.src["c"]}

    def reduce_func(self, nodes):
        h_tild = th.sum(nodes.mailbox["h"], 1)
        f = th.sigmoid(self.U_f(nodes.mailbox["h"]))
        c = th.sum(f * nodes.mailbox["c"], 1)
        return {"iou": self.U_iou(h_tild), "c": c}

    def apply_node_func(self, nodes):
        iou = nodes.data["iou"] + self.b_iou
        i, o, u = th.chunk(iou, 3, 1)
        i, o, u = th.sigmoid(i), th.sigmoid(o), th.tanh(u)
        c = i * u + nodes.data["c"]
        h = o * th.tanh(c)
        return {"h": h, "c": c}


class TreeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(TreeLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList(
            ChildSumTreeLSTMCell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        )

    def forward(self, graph, input, h=None, c=None):
        n = graph.number_of_nodes()
        h = th.zeros(
            (n, self.hidden_size * self.num_layers),
            dtype=input.dtype,
            device=input.device,
        )
        c = th.zeros(
            (n, self.hidden_size * self.num_layers),
            dtype=input.dtype,
            device=input.device,
        )

        with graph.local_scope():
            x = input
            for i, cell in enumerate(self.cells):
                graph.ndata["iou"] = cast(nn.Module, cell.W_iou)(x)
                graph.ndata["h"] = h[
                    :, (self.hidden_size * i) : (self.hidden_size * (i + 1))
                ]
                graph.ndata["c"] = c[
                    :, (self.hidden_size * i) : (self.hidden_size * (i + 1))
                ]
                dgl.prop_nodes_topo(
                    graph,
                    message_func=cell.message_func,
                    reduce_func=cell.reduce_func,
                    apply_node_func=cell.apply_node_func,
                )
                x = graph.ndata.pop("h")
            return x
