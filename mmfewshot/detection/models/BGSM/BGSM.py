import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule
from ..builder import BGSM


@BGSM.register_module()
class BGSMModule(BaseModule):
    def __init__(self, init_cfg=None):
        super(BGSMModule, self).__init__(init_cfg)

        self.threshold = 0.85

    def forward(self, features):

        aggregated = torch.sum(features, dim=1, keepdim=True)

        mean_value = torch.mean(aggregated, dim=[2, 3], keepdim=True)

        M = torch.where(aggregated >= (mean_value*self.threshold), torch.ones_like(aggregated), torch.full_like(aggregated, 0.5))

        feature_outputs = features * M.expand_as(features)

        return (feature_outputs,)
