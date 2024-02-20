__all__ = ["AttentionSum"]

import torch


class AttentionSum(torch.nn.Module):
    def __init__(self, gate_nn, feat_nn=None):
        super().__init__()
        self.gate_nn = gate_nn
        self.feat_nn = feat_nn

    def forward(self, feat, get_attention=False):
        gate = self.gate_nn(feat)
        assert gate.size(-1) == 1, "The output of gate_nn should have size 1 at the last axis."
        feat = self.feat_nn(feat) if self.feat_nn else feat

        gate = torch.nn.functional.softmax(gate, dim=1)

        r = torch.sum(torch.mul(feat, gate), dim=1)
        if get_attention:
            return r, gate
        else:
            return r

