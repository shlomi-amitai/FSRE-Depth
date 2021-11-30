from datasets.kitti_utils import *
import skimage.transform
import os
import random
import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms
import PIL.Image as pil
from utils.seg_utils import labels
from copy import deepcopy
from my_utils import *

def pil_loader(path, mode='RGB'):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    if mode == 'P':
        return Image.open(path)
    else:
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')


class UCanyonDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        filenames
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """

    def __init__(self,
                 height,
                 width,
                 frame_idxs,
                 filenames,
                 data_path='/SSD/Kitti',
                 is_train=False,
                 img_ext='.png',
                 num_scales=1,
                 ):
        super(UCanyonDataset, self).__init__()
        self.full_res_shape = (968, 608)
        self.data_path = data_path
        self.filenames = filenames
        self.height = height
        self.width = width
        self.frame_idxs = frame_idxs
        self.is_train = is_train
        self.img_ext = img_ext
        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()
        self.num_scales = num_scales

        self.brightness = (0.8, 1.2)
        self.contrast = (0.8, 1.2)
        self.saturation = (0.8, 1.2)
        self.hue = (-0.1, 0.1)
        self.K = np.array([[1215.4715960880724/self.full_res_shape[0], 0, 423.99891909157924/self.full_res_shape[0], 0],
                           [0, 1211.2257944573676/self.full_res_shape[1], 293.91172138607783/self.full_res_shape[1], 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.resize_img = {}
        self.resize_seg = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize_img[i] = transforms.Resize((self.height // s, self.width // s),
                                                   interpolation=Image.ANTIALIAS)

            self.resize_seg[i] = transforms.Resize((self.height // s, self.width // s),
                                                   interpolation=Image.BILINEAR)
        if is_train:
            self.load_depth = False

        else:
            self.load_depth = self.check_depth()

        self.full_res_shape = (968, 608)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

        # self.class_dict = self.get_classes(self.filenames)

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose seg_networks receive the
        same augmentation.
        """
        for k in list(inputs):
            if "color" in k:
                n, im, _ = k
                inputs[n[0] + '_size'] = torch.tensor(inputs[(n, im, -1)].size)
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize_img[i](inputs[(n, im, -1)])

                del inputs[(n, im, -1)]

            if "seg" in k:
                n, im, _ = k
                inputs[n[0] + '_size'] = torch.tensor(inputs[(n, im, -1)].size)
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize_seg[i](Image.fromarray(inputs[(n, im, -1)]))

                del inputs[(n, im, -1)]

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))
            elif "seg" in k:
                n, im, i = k
                inputs[(n, im, i)] = torch.tensor(np.array(f)).float().unsqueeze(0)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        inputs = {}
        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        line = self.filenames[index].split()
        folder = line[0]

        if len(line) in [3, 4, 2]:
            frame_index = int(line[1])
        else:
            frame_index = 0

        if len(line) == 3:
            side = line[2]
        else:
            side = None
        for i in self.frame_idxs:
            inputs[("color", i, -1)] = self.get_color(folder, frame_index + i, side, do_flip)

        # if side is None:
        #     if do_color_aug:
        #         color_aug = transforms.ColorJitter.get_params(
        #             self.brightness, self.contrast, self.saturation, self.hue)
        #     else:
        #         color_aug = (lambda x: x)
        #     self.preprocess(inputs, color_aug)
        #     return inputs

        inputs[("seg", 0, -1)] = self.get_depth_seg(folder, frame_index, side, do_flip)

        K = deepcopy(self.K)
        if do_flip:
            K[0, 2] = 1 - K[0, 2]

        K[0, :3] *= self.width
        K[1, :3] *= self.height
        inv_K = np.linalg.pinv(K)
        inputs[("K")] = torch.from_numpy(K)
        inputs[("inv_K")] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, side, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        return inputs

    def check_depth(self):
        # line = self.filenames[0].split()
        # scene_name = line[0]
        # frame_index = int(line[1])
        # velo_filename = os.path.join(
        #     self.data_path,
        #     scene_name,
        #     "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        # return os.path.isfile(velo_filename)
        return False

    def get_color(self, folder, frame_index, side, do_flip):
        im_path = self.get_image_path(folder, frame_index, side)

        color = self.loader(im_path)

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def get_image_path(self, folder, frame_index, side):
        idx, frameName = folder.split(',')
        idx = int(idx)
        try:
            frameName = self.filenames[idx+frame_index].split(',')[1]
        except:
            frameName = self.filenames[idx].split(',')[1]
        f_str = frameName
        image_path = os.path.join(
            self.data_path, 'imgs',
            f_str)
        return image_path

    # def get_depth(self, folder, frame_index, side, do_flip):
    #     calib_path = os.path.join(self.data_path, folder.split("/")[0])

    #     velo_filename = os.path.join(
    #         self.data_path,
    #         folder,
    #         "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

    #     depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
    #     depth_gt = skimage.transform.resize(
    #         depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

    #     if do_flip:
    #         depth_gt = np.fliplr(depth_gt)

    #     return depth_gt       

    def get_seg_map(self, folder, frame_index, side, do_flip):
        path = self.get_image_path(folder, frame_index, side)
        path = path.replace('Kitti', 'Kitti/segmentation')
        path = path.replace('/data', '')

        seg = self.loader(path, mode='P')
        seg_copy = np.array(seg.copy())

        for k in np.unique(seg):
            seg_copy[seg_copy == k] = labels[k].trainId
        seg = Image.fromarray(seg_copy, mode='P')

        if do_flip:
            seg = seg.transpose(pil.FLIP_LEFT_RIGHT)
        return seg

    def get_depth_path(self, folder, frame_index, side):
        idx, frameName = folder.split(',')
        idx = int(idx)
        try:
            frameName = self.filenames[idx+frame_index].split(',')[1]
        except:
            frameName = self.filenames[idx].split(',')[1]
        f_str = frameName
        f_str = f_str[:-5]+'_abs_depth.tif'
        image_path = os.path.join(
            self.data_path, 'depth',
            f_str)
        return image_path

    
    def get_depth_seg(self, folder, frame_index, side, do_flip):
        depth_path = self.get_depth_path(folder, frame_index, side)
        try:
            depth_gt = pil.open(depth_path)
        except:
            return None
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32)
        # depth_gt = preProcessDepth(depth_gt)

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt 