import math
import os
import pdb

import mmcv
import numpy as np
from torch.utils.data import Dataset
import dataset.transform as Xform
import torch


class DIV2K(Dataset):

    def __init__(self, data_list=None, training=False, cfg=None):
        super(DIV2K, self).__init__()
        self.cfg = cfg
        self.imgs = mmcv.list_from_file(data_list, prefix=cfg.data_root + '/')
        assert self.cfg.patch_size % self.cfg.base_resolution == 0, "Patch size must base resolution"
        self.training = training

    def __len__(self):
        if not self.cfg.debug:
            return len(self.imgs) * self.cfg.loop
        else:
            return 512

    def __getitem__(self, index_long):
        """
        :param index_long:
        :return: RGB, np.float32
        """
        index = index_long % len(self.imgs)
        img_gt = mmcv.imread(self.imgs[index]).astype(np.float32) / 255.
        if self.training:
            img_gt = Xform.random_crop(img_gt, self.cfg.patch_size)
            img_gt = Xform.augment(img_gt, hflip=self.cfg.hflip, rotation=self.cfg.rotation)
        img_gt = mmcv.bgr2rgb(img_gt)
        img_gt = torch.from_numpy(img_gt.transpose((2, 0, 1)))
        return img_gt


