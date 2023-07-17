import mmcv
import os
from glob import glob

'''
Note to fill the absolute path of your data path below
'''
dataset_dir = 'xxx/Data/DIV2K/'

origin_folder = ["DIV2K_train_HR", "DIV2K_valid_HR"]
sav_folder = ["DIV2K_train_HR_patch", "DIV2K_valid_HR_patch"]

patch_size = 480
stride = 240

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

for in_folder, out_folder in zip(origin_folder, sav_folder):
    im_list = glob(os.path.join(dataset_dir, in_folder, '*.png'))
    for index, im_path in enumerate(im_list):
        print('[*] %d/%d cropping %s ...' % (index + 1, im_list.__len__(), im_path.split('/')[-1]))
        origin = mmcv.imread(im_path)
        im_h, im_w, _ = origin.shape
        patch_num = 0
        for x in range(0, im_h - patch_size, stride):
            for y in range(0, im_w - patch_size, stride):
                patch_num += 1
                patch = origin[x:x + patch_size, y:y + patch_size, :]
                mmcv.imwrite(patch, os.path.join(dataset_dir, out_folder,
                                                 im_path.split('/')[-1].split('.')[0] + '_%03d.png' % patch_num))

train_scenes = sorted(glob(os.path.join(dataset_dir, sav_folder[0], '*.png')))
val_scenes = sorted(glob(os.path.join(dataset_dir, sav_folder[1], '*.png')))
test_scenes = sorted(glob(os.path.join(dataset_dir, 'DIV2K_valid_HR', '*.png')))
ensure_dir(os.path.join(dataset_dir, 'list'))

with open(os.path.join(dataset_dir, 'list', 'train.txt'), 'w') as f:
    for s in train_scenes:
        f.write(os.path.join(*s.split('/')[-2:]) + '\n')

with open(os.path.join(dataset_dir, 'list', 'val.txt'), 'w') as f:
    for s in val_scenes:
        f.write(os.path.join(*s.split('/')[-2:]) + '\n')

with open(os.path.join(dataset_dir, 'list', 'test.txt'), 'w') as f:
    for s in test_scenes:
        f.write(os.path.join(*s.split('/')[-2:]) + '\n')
