__all__ = ["GlycoSpectralAngleLoss"]


from typing import Dict, cast

import torch
from torch.nn.modules.loss import _Loss

from ...util.math import spectral_angle
from ..common.data import GlycoPeptideDataBatch
from .data import (
    GlycoPeptideMS2DataBatch,
    GlycoPeptideMS2Output,
    adjust_fragment_intensity_by_ratio,
    unbatch_glycan_fragment_intensity,
)


class GlycoSpectralAngleLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean"):
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)

    def forward(
        self, batch: GlycoPeptideMS2DataBatch, pred: GlycoPeptideMS2Output
    ) -> Dict[str, torch.Tensor]:
        pep_pred, gly_pred, ratio_pred = (
            pred.peptide_fragment_intensity,
            pred.glycan_fragment_intensity,
            pred.fragment_intensity_ratio,
        )
        pep_loss = spectral_angle(pep_pred, batch.peptide_fragment_intensity)

        gly_pred_actual = unbatch_glycan_fragment_intensity(
            cast(GlycoPeptideDataBatch, batch),
            gly_pred,
            batch.glycan_fragment_intensity,
            recover_order=False,
        )
        gly_loss = torch.concat(
            [
                spectral_angle(g_pred, g_actual, batched=False).unsqueeze(0)
                for g_pred, g_actual in gly_pred_actual
            ]
        )
        gp_loss = torch.concat(
            [
                spectral_angle(
                    torch.concat(
                        adjust_fragment_intensity_by_ratio(
                            pep_pred[i].flatten(),
                            g_pred.flatten(),
                            ratio_pred[i],
                        ),
                        dim=-1,
                    ),
                    torch.concat(
                        adjust_fragment_intensity_by_ratio(
                            batch.peptide_fragment_intensity[i].flatten(),
                            g_actual.flatten(),
                            batch.fragment_intensity_ratio[i],
                        ),
                        dim=-1,
                    ),
                    batched=False,
                ).unsqueeze(0)
                for i, (g_pred, g_actual) in enumerate(gly_pred_actual)
            ]
        )

        sa = {
            "sa_total": gp_loss,
            "sa_pep": pep_loss,
            "sa_gly": gly_loss,
        }
        if self.reduction == "mean":
            return {k: v.mean() for k, v in sa.items()}
        elif self.reduction == "sum":
            return {k: v.sum() for k, v in sa.items()}
        else:
            return sa
