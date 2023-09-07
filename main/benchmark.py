#!/usr/bin/env python


import os
import cv2
from os.path import join
from tqdm import tqdm
from base.utilities import get_parser, get_logger, AverageMeter

cfg = get_parser()
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in cfg.test_gpu)
import torch
print(torch.cuda.device_count())

from models import get_model
import numpy as np
from utils import util
from base.baseTrainer import load_state_dict
import mmcv
from dataset.torch_bicubic import imresize
import math
cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

## whether or not crop border of the generated images
crop = True

## whether or not adopt lpips as one of the evaluation metrics
benchmark_lpips = False
if benchmark_lpips:
    import lpips
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()

## optionally record the quantitative results
import csv


def main():
    global cfg, logger
    logger = get_logger()
    logger.info(cfg)
    logger.info("=> creating model ...")
    model = get_model(cfg, logger)
    model = model.cuda()
    model.summary(logger, None)

    if os.path.isfile(cfg.model_path):
        logger.info("=> loading checkpoint '{}'".format(cfg.model_path))
        checkpoint = torch.load(cfg.model_path, map_location=lambda storage, loc: storage.cpu())
        load_state_dict(model, checkpoint['state_dict'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(cfg.model_path, checkpoint['epoch']))
    else:
        raise RuntimeError("=> no checkpoint flound at '{}'".format(cfg.model_path))

    # ####################### Data Loader ####################### #
    data_names = cfg.test_dataset.split('+') if '+' in cfg.test_dataset else [cfg.test_dataset]
    for data_name in data_names:
        cfg.data_name = data_name
        cfg.data_root = cfg.test_root
        if cfg.data_name in ['Set5', 'Set14', 'BSDS100', 'urban100', 'DIV2K', 'DIV2K_valid_HR_patch']:
            from dataset.div2k import DIV2K
            test_data = DIV2K(data_list=os.path.join(cfg.test_root, 'list', data_name+'_val.txt'), training=False,
                            cfg=cfg) if cfg.evaluate else None
        
        else:
            raise Exception('Dataset not supported yet'.format(cfg.data_name))
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.test_batch_size,
                                                shuffle=False, num_workers=cfg.workers, pin_memory=True,
                                                drop_last=False)
        test_scales = cfg.test_scale.split('+') if '+' in str(cfg.test_scale) else [cfg.test_scale]
        for scale in test_scales:
            logger.info("\n=> Dataset '{}' (x{})\n".format(data_name, scale))
            test(model, test_loader, scale=float(scale), save=cfg.save, data_name=cfg.data_name)


def test(model, test_data_loader, scale=4, save=False, data_name=None):
    psnr_meter_lr, psnr_meter_hr = AverageMeter(), AverageMeter()
    ssim_meter_lr, ssim_meter_hr = AverageMeter(), AverageMeter()

    psnr_y_meter_lr, psnr_y_meter_hr = AverageMeter(), AverageMeter()
    ssim_y_meter_lr, ssim_y_meter_hr = AverageMeter(), AverageMeter()
    filepath = os.path.join(cfg.save_folder, cfg.data_name +'x'+str(scale)+'.txt')
    with torch.no_grad():
        model.eval()
        with open(filepath,'w') as f:
            for i, hr in enumerate(tqdm(test_data_loader)):
                hr = hr.cuda()
                lr = imresize(hr, scale=1.0 / scale).detach()
                encoded_lr, restored_hr = model(hr, scale)

                encoded_lr = torch.clamp(encoded_lr, 0, 1)
                restored_hr = torch.clamp(restored_hr, 0, 1)

                ########################## CALCULATE METRIC
                encoded_lr = util.tensor2img(encoded_lr)
                restored_hr = util.tensor2img(restored_hr)
                lr = util.tensor2img(lr)
                hr = util.tensor2img(hr)
                if crop:
                    crop_border = math.ceil(scale)
                    encoded_lr = encoded_lr[crop_border:-crop_border, crop_border:-crop_border, :]
                    restored_hr = restored_hr[crop_border:-crop_border, crop_border:-crop_border, :]
                    lr = lr[crop_border:-crop_border, crop_border:-crop_border, :]
                    hr = hr[crop_border:-crop_border, crop_border:-crop_border, :]

                psnr_meter_lr.update(util.calculate_psnr(encoded_lr*255, lr*255))
                ssim_meter_lr.update(util.calculate_ssim(encoded_lr*255, lr*255))

                psnr_meter_hr.update(util.calculate_psnr(restored_hr*255, hr*255))
                ssim_meter_hr.update(util.calculate_ssim(restored_hr*255, hr*255))

                if benchmark_lpips:
                    lpips_a = loss_fn_alex(torch.from_numpy(restored_hr*2-1.0).cuda()[None].permute(0,3,1,2).float(), torch.from_numpy(hr*2-1.0).cuda()[None].permute(0,3,1,2).float())
                    lpips_a = lpips_a.cpu().detach().item()
                else:
                    lpips_a = 0
                # calculate metric on Y channel
                encoded_lr_y = util.rgb2ycbcr(encoded_lr, only_y=True)
                restored_hr_y = util.rgb2ycbcr(restored_hr, only_y=True)
                lr_y = util.rgb2ycbcr(lr, only_y=True)
                hr_y = util.rgb2ycbcr(hr, only_y=True)


                psnr_y_lr_ = util.calculate_psnr(encoded_lr_y*255, lr_y*255)
                ssim_y_lr_ = util.calculate_ssim(encoded_lr_y*255, lr_y*255)
                psnr_y_meter_lr.update(psnr_y_lr_)
                ssim_y_meter_lr.update(ssim_y_lr_)

                psnr_y_hr_ = util.calculate_psnr(restored_hr_y*255, hr_y*255)
                ssim_y_hr_ = util.calculate_ssim(restored_hr_y*255, hr_y*255)
                psnr_y_meter_hr.update(psnr_y_hr_)
                ssim_y_meter_hr.update(ssim_y_hr_)
                # print("-> {} PSNR-Y: {:.2f}  SSIM-Y: {:.4f}  LPIPS: {:.4f}".format(test_data_loader.dataset.imgs[i].split('/')[-1].split('.')[0], psnr_y_hr_, ssim_y_hr_, lpips_a))
                
                if save:
                    #f.write(test_data_loader.dataset.imgs[i].split('/')[-1].split('.')[0]+' '+'{:.2f}'.format(psnr_y_hr_)+' '+ '{:.4f}'.format(ssim_y_hr_) + ' ' +'{:.4f}'.format(lpips_a) +'_'+'{:.2f}'.format(psnr_y_lr_)+' '+ '{:.4f}'.format(ssim_y_lr_) +'\n')
                    for x, last_fix in zip([encoded_lr, restored_hr],
                                        ["_res_lr.png", "_res_sr.png"]):
                        mmcv.imwrite(mmcv.rgb2bgr(x),
                                    join(cfg.save_folder, cfg.data_name + '_x%.1f' % scale,
                                        test_data_loader.dataset.imgs[i].split('/')[-1].split('.')[0] + last_fix))
    logger.info('Crop:{crop}\n==>res_lr: \n'
                'PSNR: {psnr_meter_lr.avg:.2f}\n'
                'SSIM: {ssim_meter_lr.avg:.4f}\n'
                'PSNR-Y: {psnr_y_meter_lr.avg:.2f}\n'
                'SSIM-Y: {ssim_y_meter_lr.avg:.4f}\n'
                '==>res_sr: \n'
                'PSNR: {psnr_meter_hr.avg:.2f}\n'
                'SSIM: {ssim_meter_hr.avg:.4f}\n'
                'PSNR-Y: {psnr_y_meter_hr.avg:.2f}\n'
                'SSIM-Y: {ssim_y_meter_hr.avg:.4f}\n'.format(crop=crop, psnr_meter_lr=psnr_meter_lr, ssim_meter_lr=ssim_meter_lr,
                                                        psnr_y_meter_lr=psnr_y_meter_lr, ssim_y_meter_lr=ssim_y_meter_lr,
                                                         psnr_meter_hr=psnr_meter_hr, ssim_meter_hr=ssim_meter_hr,
                                                         psnr_y_meter_hr=psnr_y_meter_hr, ssim_y_meter_hr=ssim_y_meter_hr
                                                         ))


if __name__ == '__main__':
    main()
