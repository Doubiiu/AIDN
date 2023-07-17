# Standard libraries
import itertools
import numpy as np
# PyTorch
import torch
import torch.nn as nn
# Local
import models.lib.utils as utils
import torch.nn.functional as F


class rgb_to_ycbcr_jpeg(nn.Module):
    """ Converts RGB image to YCbCr
    Input:
        image(tensor): batch x 3 x height x width
    Outpput:
        result(tensor): batch x height x width x 3
    """
    def __init__(self):
        super(rgb_to_ycbcr_jpeg, self).__init__()
        matrix = np.array(
            [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5],
             [0.5, -0.418688, -0.081312]], dtype=np.float32).T
        self.shift = torch.tensor([0., 128., 128.])
        self.matrix = torch.from_numpy(matrix)

    def forward(self, image):
        image = image.permute(0, 2, 3, 1)
        if image.is_cuda:
            self.shift = self.shift.cuda(image.get_device())
            self.matrix = self.matrix.cuda(image.get_device())
        result = torch.tensordot(image, self.matrix, dims=1) + self.shift
        result.view(image.shape)
        return result



class chroma_subsampling(nn.Module):
    """ Chroma subsampling on CbCv channels
    Input:
        image(tensor): batch x height x width x 3
    Output:
        y(tensor): batch x height x width
        cb(tensor): batch x height/2 x width/2
        cr(tensor): batch x height/2 x width/2
    """
    def __init__(self):
        super(chroma_subsampling, self).__init__()

    def forward(self, image):
        image_2 = image.permute(0, 3, 1, 2).clone()
        avg_pool = nn.AvgPool2d(kernel_size=2, stride=(2, 2), ceil_mode=True,
                                count_include_pad=False)
        cb = avg_pool(image_2[:, 1, :, :].unsqueeze(1))
        cr = avg_pool(image_2[:, 2, :, :].unsqueeze(1))
        cb = cb.permute(0, 2, 3, 1)
        cr = cr.permute(0, 2, 3, 1)
        return image[:, :, :, 0], cb.squeeze(3), cr.squeeze(3)


def repeat(x,h_padding,w_padding):
    last_h = x[...,-1,:]
    last_h = last_h.unsqueeze(-2)

    hs = last_h.expand(-1,-1,h_padding,-1)
    x = torch.cat([x,hs], dim=2)
    last_w = x[...,-1]
    last_w = last_w.unsqueeze(-1)

    ws = last_w.expand(-1,-1,-1,w_padding)

    y = torch.cat([x,ws], dim=-1)
    return y.contiguous()

class block_splitting(nn.Module):
    """ Splitting image into patches
    Input:
        image(tensor): batch x height x width
    Output: 
        patch(tensor):  batch x h*w/64 x h x w
    """
    def __init__(self):
        super(block_splitting, self).__init__()
        self.k = 8

    def forward(self, image):
        image = image.unsqueeze(1)
        hp = 8-image.shape[2]%8 if image.shape[2]%8 != 0 else 0
        wp = 8-image.shape[3]%8 if image.shape[3]%8 != 0 else 0
        image = repeat(image, hp, wp)

        img_unf = F.unfold(image, kernel_size=(self.k, self.k),
                           stride=(self.k, self.k))
        unf_shape = img_unf.shape

        img_unf = img_unf.view([unf_shape[0], self.k, self.k, unf_shape[2]])

        return img_unf.contiguous().permute(0, 3, 1, 2).contiguous()
    

class dct_8x8(nn.Module):
    """ Discrete Cosine Transformation
    Input:
        image(tensor): batch x height x width
    Output:
        dcp(tensor): batch x height x width
    """
    def __init__(self):
        super(dct_8x8, self).__init__()
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
                (2 * y + 1) * v * np.pi / 16)
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        #
        self.tensor =  torch.from_numpy(tensor).float()
        self.scale = torch.from_numpy(np.outer(alpha, alpha) * 0.25).float()
        
    def forward(self, image):
        if image.is_cuda:
            self.tensor = self.tensor.cuda(image.get_device())
            self.scale = self.scale.cuda(image.get_device())
        image = image - 128
        result = self.scale * torch.tensordot(image, self.tensor, dims=2)
        result.view(image.shape)
        return result


class y_quantize(nn.Module):
    """ JPEG Quantization for Y channel
    Input:
        image(tensor): batch x height x width
        rounding(function): rounding function to use
        factor(float): Degree of compression
    Output:
        image(tensor): batch x height x width
    """
    def __init__(self, rounding, factor=1):
        super(y_quantize, self).__init__()
        self.rounding = rounding
        self.factor = factor
        self.y_table = utils.y_table

    def forward(self, image):
        if image.is_cuda:
            self.y_table = self.y_table.cuda(image.get_device())

        image = image.float() / (self.y_table * self.factor)
        image = self.rounding(image)

        return image


class c_quantize(nn.Module):
    """ JPEG Quantization for CrCb channels
    Input:
        image(tensor): batch x height x width
        rounding(function): rounding function to use
        factor(float): Degree of compression
    Output:
        image(tensor): batch x height x width
    """
    def __init__(self, rounding, factor=1):
        super(c_quantize, self).__init__()
        self.rounding = rounding
        self.factor = factor
        self.c_table = utils.c_table

    def forward(self, image):
        if image.is_cuda:
            self.c_table = self.c_table.cuda(image.get_device())
        image = image.float() / (self.c_table * self.factor)
        image = self.rounding(image)
        return image


class compress_jpeg(nn.Module):
    """ Full JPEG compression algortihm
    Input:
        imgs(tensor): batch x 3 x height x width
        rounding(function): rounding function to use
        factor(float): Compression factor
    Ouput:
        compressed(dict(tensor)): batch x h*w/64 x 8 x 8
    """
    def __init__(self, rounding=torch.round, factor=1):
        super(compress_jpeg, self).__init__()
        self.l1 = nn.Sequential(
            rgb_to_ycbcr_jpeg(),
            chroma_subsampling()
        )
        self.l2 = nn.Sequential(
            block_splitting(),
            dct_8x8()
        )
        self.c_quantize = c_quantize(rounding=rounding, factor=factor)
        self.y_quantize = y_quantize(rounding=rounding, factor=factor)

    def forward(self, image):
        y, cb, cr = self.l1(image*255)

        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            comp = self.l2(components[k])
            if k in ('cb', 'cr'):
                comp = self.c_quantize(comp)
            else:
                comp = self.y_quantize(comp)

            components[k] = comp
        return components['y'], components['cb'], components['cr']
