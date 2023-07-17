import torch
import torch.nn as nn
import math

class Quant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        input = torch.clamp(input, 0, 1)
        output = (input * 255.).round() / 255.
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class Quantization(nn.Module):
    def __init__(self):
        super(Quantization, self).__init__()

    def forward(self, input):
        return Quant.apply(input)

class Quant_RS(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        
        input = torch.clamp(input, 0, 1)
        ctx.save_for_backward(input)
        output = (input * 255.).round() / 255.
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output*(1-0.5*torch.cos(2*math.pi*input))


class Quantization_RS(nn.Module):
    def __init__(self):
        super(Quantization_RS, self).__init__()

    def forward(self, input):
        return Quant_RS.apply(input)