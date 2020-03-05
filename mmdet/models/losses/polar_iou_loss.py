import torch.nn as nn
from ..registry import LOSSES
import torch


@LOSSES.register_module
class PolarIOULoss(nn.Module):
    def __init__(self):
        super(PolarIOULoss, self).__init__()

    def forward(self, pred, target, weight, avg_factor=None):
        """
         :param pred:  shape (N,36), N is nr_box
         :param target: shape (N,36)
         :return: loss

        Args:

            avg_factor:
            weight:
         """

        total = torch.stack([pred, target], -1)
        l_max = total.max(dim=2)[0]
        l_min = total.min(dim=2)[0]

        loss = (l_max.sum(dim=1) / l_min.sum(dim=1)).log()
        loss = loss * weight
        loss = loss.sum() / avg_factor
        return loss
