from base.base_model import BaseModel
import torch.nn as nn
from models.common import MeanShift, default_conv, ResBlock, Upsampler, Downsampler
from models.arb import SA_adapt, SCAB_upsample, SCAB_downsample


class EDRS(BaseModel):
    def __init__(self, args=None):
        super(EDRS, self).__init__()
        n_resblocks = args.n_resblocks
        self.n_resblocks = n_resblocks
        n_feats = args.n_feats
        # scale = args.scale
        kernel_size = 3
        act = nn.ReLU(True)
        conv = default_conv
        rescale = args.rescale
        self.rescale = rescale
        self.fixed_scale = args.fixed_scale

        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        if not args.fixed_scale:
            m_tail = [
                # Upsampler(conv, scale, n_feats, act=False),
                None,
                conv(n_feats, args.n_colors, kernel_size)
            ] if rescale == 'up' else [
                # Downsampler(conv, scale, n_feats, act=False),
                None,
                conv(n_feats, args.n_colors, kernel_size)
            ]
        else:
            m_tail = [
                Upsampler(conv, args.scale, n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size)
            ] if rescale == 'up' else [
                Downsampler(conv, args.scale, n_feats, act=False),
                conv(n_feats, args.n_colors, kernel_size)
            ]
        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

        if not args.fixed_scale:
            # For EDSR backbone, we set K=4
            self.K = args.K
            sa_adapt = []
            for i in range(self.n_resblocks // self.K):
                sa_adapt.append(SA_adapt(channels=args.n_feats, num_experts=args.num_experts_SAconv))
            self.sa_adapt = nn.Sequential(*sa_adapt)

            # Scale-aware Downsample and Upsample
            if rescale == 'up':
                if args.up_sampler == 'sampleB':
                    self.sa_sample = SCAB_upsample(channels=args.n_feats, num_experts=args.num_experts_CRM)

            else:
                if args.down_sampler == 'sampleB':
                    self.sa_sample = SCAB_downsample(channels=args.n_feats, num_experts=args.num_experts_CRM)




    def forward(self, x, scale, outH=None, outW=None):
        x = self.sub_mean(x)
        x = self.head(x)

        #body
        if not self.fixed_scale:
            res = x
            for i in range(self.n_resblocks):
                res = self.body[i](res)
                # scale modulation 
                if (i+1)% self.K == 0:
                    res = self.sa_adapt[(i+1) // self.K -1](res, scale)

            res = self.body[-1](res)
            res += x

            
            if self.rescale == 'up':
                res = self.sa_sample(res, scale, outH, outW)
            else:
                res = self.sa_sample(res, scale)
            x = self.tail[1](res)
        else:
            res = self.body(x)
            res += x

            x = self.tail(res)

        x = self.add_mean(x)
        return x