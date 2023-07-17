#!/usr/bin/env python
import torch.nn.functional as F
import torch
import torch.nn as nn


def grad_l1_loss(y_input, y_target):
    inputGradH = y_input[..., 1:, :] - y_input[..., :-1, :]
    inputGradW = y_input[..., :, 1:] - y_input[..., :, :-1]
    targetGradH = y_target[..., 1:, :] - y_target[..., :-1, :]
    targetGradW = y_target[..., :, 1:] - y_target[..., :, :-1]
    hLoss = F.l1_loss(inputGradH, targetGradH)
    wLoss = F.l1_loss(inputGradW, targetGradW)
    return (hLoss + wLoss) / 2.


class GradSoftL1(nn.Module):
    def __init__(self, ):
        super(GradSoftL1, self).__init__()
        self.l1_fun = CharbonnierLoss()

    def forward(self, y_input, y_target):
        inputGradH = y_input[..., 1:, :] - y_input[..., :-1, :]
        inputGradW = y_input[..., :, 1:] - y_input[..., :, :-1]
        targetGradH = y_target[..., 1:, :] - y_target[..., :-1, :]
        targetGradW = y_target[..., :, 1:] - y_target[..., :, :-1]
        hLoss = self.l1_fun(inputGradH, targetGradH)
        wLoss = self.l1_fun(inputGradW, targetGradW)
        return (hLoss + wLoss) / 2.


class CharbonnierLoss(nn.Module):
    """L1 Charbonnierloss."""

    def __init__(self):
        super(CharbonnierLoss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
