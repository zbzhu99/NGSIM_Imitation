from torch import nn
import torch
import torch.nn.functional as F


class BCEFocalLoss(nn.Module):
    def __init__(self, gamma):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        targets = targets.long().squeeze()
        prob_1 = torch.sigmoid(inputs)
        prob_0 = 1.0 - prob_1
        soft_inputs = torch.concat([prob_0, prob_1], dim=-1)
        target_one_hot = F.one_hot(targets, num_classes=2)
        weight = torch.pow(1.0 - soft_inputs, self.gamma)
        focal = -weight * torch.log(soft_inputs)
        assert target_one_hot.shape == focal.shape
        loss = torch.sum(target_one_hot * focal, dim=-1).mean()
        return loss
