
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import math


def is_pytorch_version_higher_than(target_version="1.3.0"):
    installed_version = torch.__version__
    installed_version_tuple = tuple(map(int, (installed_version.split("+")[0]).split(".")))
    target_version_tuple = tuple(map(int, target_version.split(".")))
    return installed_version_tuple >= target_version_tuple

class SCAB_downsample(nn.Module):
    def __init__(self, channels=64, num_experts=4, bias=False):
        super(SCAB_downsample, self).__init__()
        self.bias = bias
        self.num_experts = num_experts
        self.channels = channels

        # experts
        weight_compress = []
        for i in range(num_experts):
            weight_compress.append(nn.Parameter(torch.Tensor(channels//8, channels, 1, 1)))
            nn.init.kaiming_uniform_(weight_compress[i], a=math.sqrt(5))
        self.weight_compress = nn.Parameter(torch.stack(weight_compress, 0))

        weight_expand = []
        for i in range(num_experts):
            weight_expand.append(nn.Parameter(torch.Tensor(channels, channels//8, 1, 1)))
            nn.init.kaiming_uniform_(weight_expand[i], a=math.sqrt(5))
        self.weight_expand = nn.Parameter(torch.stack(weight_expand, 0))

        # two FC layers
        self.body = nn.Sequential(
            nn.Conv2d(self.channels+4, self.channels, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(self.channels, self.channels, 1, 1, 0, bias=True),
            nn.ReLU(True),
        )
        # routing head
        self.routing = nn.Sequential(
            nn.Conv2d(self.channels, num_experts, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )
        # offset head
        self.offset = nn.Conv2d(self.channels, 2, 1, 1, 0, bias=True)

    def forward(self, x, scale):
        b, c, h, w = x.size()
 
        # (1) coordinates in LR space
        ## coordinates in HR space
        coor_hr = [torch.arange(0, math.ceil(h * scale), 1).unsqueeze(0).float().to(x.device),
                   torch.arange(0, math.ceil(w * scale), 1).unsqueeze(0).float().to(x.device)]

        ## coordinates in LR space
        coor_h = ((coor_hr[0] + 0.5) / scale) - (torch.floor((coor_hr[0] + 0.5) / scale + 1e-3)) - 0.5
        coor_h = coor_h.permute(1, 0)
        coor_w = ((coor_hr[1] + 0.5) / scale) - (torch.floor((coor_hr[1] + 0.5) / scale + 1e-3)) - 0.5

        input = torch.cat((
            torch.ones_like(coor_h).expand([-1, math.ceil(scale * w)]).unsqueeze(0) / scale,
            torch.ones_like(coor_h).expand([-1, math.ceil(scale * w)]).unsqueeze(0) / scale,
            coor_h.expand([-1, math.ceil(scale * w)]).unsqueeze(0),
            coor_w.expand([math.ceil(scale * h), -1]).unsqueeze(0)
        ), 0).unsqueeze(0)
        pre_fea = grid_sample(x, None, scale) #b 64 h w
        input = torch.cat([input.expand([b, -1, -1, -1]), pre_fea], dim=1)


        # (2) predict filters and offsets
        embedding = self.body(input) # b 64 h w
        ## offsets
        offset = self.offset(embedding)


        # (3) grid sample & spatially varying filtering
        ## grid sample
        fea0 = grid_sample(x, offset, scale)               ## b c h w
        fea = fea0.unsqueeze(-1).permute(0, 2, 3, 1, 4)            ## b c h w 1 -> b * h * w * c * 1


        ## filters: improvement content-aware
        routing_weights = self.routing(embedding) # b 4 h w
        routing_weights = routing_weights.view(b, self.num_experts, math.ceil(scale*h) * math.ceil(scale*w)).permute(0, 2, 1)      # (h*w) * n

        weight_compress = self.weight_compress.view(self.num_experts, -1) # 4 c/8 c 1 1 -> 4 c/8*c
        weight_compress = torch.matmul(routing_weights, weight_compress) # b (h*w) 4 matmul 4 c/8*c -> b h*w c/8*c
        weight_compress = weight_compress.view(b, math.ceil(scale*h), math.ceil(scale*w), self.channels//8, self.channels)# b h w c/8 c

        weight_expand = self.weight_expand.view(self.num_experts, -1)
        weight_expand = torch.matmul(routing_weights, weight_expand)
        weight_expand = weight_expand.view(b, math.ceil(scale*h), math.ceil(scale*w), self.channels, self.channels//8)

        ## spatially varying filtering
        out = torch.matmul(weight_compress, fea) # b h w c/8 c * b h w c 1 = b h w c/8 1
        out = torch.matmul(weight_expand, out).squeeze(-1) # b h w c c/8 * b h w c/8 1 = b h w c 1

        return out.permute(0, 3, 1, 2) + fea0 # b c h w + b c h w


class SCAB_upsample(nn.Module):
    def __init__(self, channels=64, num_experts=4, bias=False):
        super(SCAB_upsample, self).__init__()
        self.bias = bias
        self.num_experts = num_experts
        self.channels = channels

        # experts
        weight_compress = []
        for i in range(num_experts):
            weight_compress.append(nn.Parameter(torch.Tensor(channels//8, channels, 1, 1)))
            nn.init.kaiming_uniform_(weight_compress[i], a=math.sqrt(5))
        self.weight_compress = nn.Parameter(torch.stack(weight_compress, 0))

        weight_expand = []
        for i in range(num_experts):
            weight_expand.append(nn.Parameter(torch.Tensor(channels, channels//8, 1, 1)))
            nn.init.kaiming_uniform_(weight_expand[i], a=math.sqrt(5))
        self.weight_expand = nn.Parameter(torch.stack(weight_expand, 0))

        # two FC layers
        self.body = nn.Sequential(
            nn.Conv2d(self.channels+4, self.channels, 1, 1, 0, bias=True),
            nn.ReLU(True),
            nn.Conv2d(self.channels, self.channels, 1, 1, 0, bias=True),
            nn.ReLU(True),
        )
        # routing head
        self.routing = nn.Sequential(
            nn.Conv2d(self.channels, num_experts, 1, 1, 0, bias=True),
            nn.Sigmoid()
        )
        # offset head
        self.offset = nn.Conv2d(self.channels, 2, 1, 1, 0, bias=True)

    def forward(self, x, scale, outH, outW):
        b, c, h, w = x.size()

        # (1) coordinates in LR space
        ## coordinates in HR space
        coor_hr = [torch.arange(0, outH, 1).unsqueeze(0).float().to(x.device),
                   torch.arange(0, outW, 1).unsqueeze(0).float().to(x.device)]

        ## coordinates in LR space
        coor_h = ((coor_hr[0] + 0.5) / scale) - (torch.floor((coor_hr[0] + 0.5) / scale + 1e-3)) - 0.5
        coor_h = coor_h.permute(1, 0)
        coor_w = ((coor_hr[1] + 0.5) / scale) - (torch.floor((coor_hr[1] + 0.5) / scale + 1e-3)) - 0.5

        input = torch.cat((
            torch.ones_like(coor_h).expand([-1, outW]).unsqueeze(0) / scale,
            torch.ones_like(coor_h).expand([-1, outW]).unsqueeze(0) / scale,
            coor_h.expand([-1, outW]).unsqueeze(0),
            coor_w.expand([outH, -1]).unsqueeze(0)
        ), 0).unsqueeze(0)
        pre_fea = grid_sample(x, None, scale, outH, outW) #b 64 h w

        input = torch.cat([input.expand([b, -1, -1, -1]), pre_fea], dim=1)

        # (2) predict filters and offsets
        embedding = self.body(input)
        ## offsets
        offset = self.offset(embedding)

        # (3) grid sample & spatially varying filtering
        fea0 = grid_sample(x, offset, scale, outH, outW)               ## b * h * w * c * 1
        fea = fea0.unsqueeze(-1).permute(0, 2, 3, 1, 4)            ## b * h * w * c * 1


        ## filters
        routing_weights = self.routing(embedding)
        routing_weights = routing_weights.view(b, self.num_experts, outH * outW).permute(0, 2, 1)      # (h*w) * n

        ##
        weight_compress = self.weight_compress.view(self.num_experts, -1)
        weight_compress = torch.matmul(routing_weights, weight_compress)
        weight_compress = weight_compress.view(b, outH, outW, self.channels//8, self.channels)

        weight_expand = self.weight_expand.view(self.num_experts, -1)
        weight_expand = torch.matmul(routing_weights, weight_expand)
        weight_expand = weight_expand.view(b, outH, outW, self.channels, self.channels//8)

        ## spatially varying filtering
        out = torch.matmul(weight_compress, fea)
        out = torch.matmul(weight_expand, out).squeeze(-1)

        return out.permute(0, 3, 1, 2) + fea0


class SA_adapt(nn.Module):
    def __init__(self, channels, num_experts=4):
        super(SA_adapt, self).__init__()
        self.mask = nn.Sequential(
            nn.Conv2d(channels, channels//4, 3, 1, 1),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(True),
            nn.Conv2d(channels//4, channels//4, 3, 1, 1),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(True),
            nn.Conv2d(channels//4, channels//4, 3, 1, 1),
            nn.BatchNorm2d(channels//4),
            nn.ReLU(True),
            nn.Conv2d(channels//4, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.adapt = SA_conv(channels, channels, 3, 1, 1, num_experts=num_experts)

    def forward(self, x, scale):
        mask = self.mask(x)
        adapted = self.adapt(x, scale)
        
        return x + adapted * mask


class SA_conv(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size=3, stride=1, padding=1, bias=False, num_experts=4):
        super(SA_conv, self).__init__()
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.num_experts = num_experts
        self.bias = bias

        # FC layers to generate routing weights
        self.routing = nn.Sequential(
            nn.Linear(2, channels_in),
            nn.ReLU(True),
            nn.Linear(channels_in, num_experts),
            nn.Softmax(1)
        )

        # initialize experts
        weight_pool = []
        for i in range(num_experts):
            weight_pool.append(nn.Parameter(torch.Tensor(channels_out, channels_in, kernel_size, kernel_size)))
            nn.init.kaiming_uniform_(weight_pool[i], a=math.sqrt(5))
        self.weight_pool = nn.Parameter(torch.stack(weight_pool, 0))

        if bias:
            self.bias_pool = nn.Parameter(torch.Tensor(num_experts, channels_out))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_pool)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_pool, -bound, bound)

    def forward(self, x, scale):
        # generate routing weights
        scale = torch.ones(1, 1).to(x.device) / scale

        routing_weights = self.routing(torch.cat((scale, scale), 1)).view(self.num_experts, 1, 1)

        # fuse experts
        fused_weight = (self.weight_pool.view(self.num_experts, -1, 1) * routing_weights).sum(0)
        fused_weight = fused_weight.view(-1, self.channels_in, self.kernel_size, self.kernel_size)

        if self.bias:
            fused_bias = torch.mm(routing_weights, self.bias_pool).view(-1)
        else:
            fused_bias = None

        # convolution
        out = F.conv2d(x, fused_weight, fused_bias, stride=self.stride, padding=self.padding)

        return out


def grid_sample(x, offset, scale, outH=None, outW=None):
    # generate grids
    b, _, h, w = x.size()
    if outH is None:
        grid = np.meshgrid(range(math.ceil(scale*w)), range(math.ceil(scale*h)))
    else:
        grid = np.meshgrid(range(outW), range(outH))
    grid = np.stack(grid, axis=-1).astype(np.float64)
    grid = torch.Tensor(grid).to(x.device)

    # project into LR space
    grid[:, :, 0] = (grid[:, :, 0] + 0.5) / scale - 0.5
    grid[:, :, 1] = (grid[:, :, 1] + 0.5) / scale - 0.5

    # normalize to [-1, 1]
    grid[:, :, 0] = grid[:, :, 0] * 2 / (w - 1) -1
    grid[:, :, 1] = grid[:, :, 1] * 2 / (h - 1) -1
    grid = grid.permute(2, 0, 1).unsqueeze(0)
    grid = grid.expand([b, -1, -1, -1])

    # add offsets
    if offset is not None:
        offset_0 = torch.unsqueeze(offset[:, 0, :, :] * 2 / (w - 1), dim=1)
        offset_1 = torch.unsqueeze(offset[:, 1, :, :] * 2 / (h - 1), dim=1)
        grid = grid + torch.cat((offset_0, offset_1),1)
    grid = grid.permute(0, 2, 3, 1)

    ## sampling
    ## Remember to add "align_corners=True" to F.grid_sample() if you are using pytorch>=1.3.0
    # if is_pytorch_version_higher_than(target_version="1.3.0"):
    #     output = F.grid_sample(x, grid, padding_mode='zeros', align_corners=True)
    # else:
    output = F.grid_sample(x, grid, padding_mode='zeros')

    return output