import torch
import torch.nn as nn
# Local
from .modules import compress_jpeg, decompress_jpeg
from .utils import diff_round, quality_to_factor


class JPGQuantizeFun(nn.Module):
    def __init__(self, differentiable=True, quality=90):
        ''' Initialize the DiffJPEG layer
        Inputs:
            height(int): Original image hieght
            width(int): Original image width
            differentiable(bool): If true uses custom differentiable
                rounding function, if false uses standrard torch.round
            quality(float): Quality factor for jpeg compression scheme. 
        '''
        super(JPGQuantizeFun, self).__init__()
        if differentiable:
            rounding = diff_round
        else:
            rounding = torch.round
        factor = quality_to_factor(quality)
        self.compress = compress_jpeg(rounding=rounding, factor=factor)
        self.decompress = decompress_jpeg(rounding=rounding, factor=factor)

    def forward(self, x):
        '''
        '''
        _, _, h, w = x.shape
        y, cb, cr = self.compress(x)
        recovered = self.decompress(y, cb, cr, h, w)
        return recovered