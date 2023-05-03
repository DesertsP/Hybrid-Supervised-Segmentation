import logging

import numpy as np
import torch
from torch.utils.data import Dataset
import os.path
from utils import visualize
from PIL import Image, ImagePalette
import torchvision.transforms as tf
import torchvision.transforms.functional as ttf

IMG_FOLDER_NAME = "JPEGImages"
ANNOT_FOLDER_NAME = "Annotations"
SEGMENTATION_FOLDER_NAME = 'SegmentationClass'
IGNORE = 255

CAT_LIST = ['aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']

N_CAT = len(CAT_LIST)

CAT_NAME_TO_NUM = dict(zip(CAT_LIST, range(len(CAT_LIST))))


def decode_int_filename(int_filename):
    s = str(int(int_filename))
    return s[:4] + '_' + s[4:]


def load_image_label_from_xml(img_name, voc12_root):
    from xml.dom import minidom

    elem_list = minidom.parse(
        os.path.join(voc12_root, ANNOT_FOLDER_NAME, decode_int_filename(img_name) + '.xml')).getElementsByTagName(
        'name')

    multi_cls_lab = np.zeros((N_CAT), np.float32)

    for elem in elem_list:
        cat_name = elem.firstChild.data
        if cat_name in CAT_LIST:
            cat_num = CAT_NAME_TO_NUM[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab


def load_image_label_list_from_xml(img_name_list, voc12_root):
    return [load_image_label_from_xml(img_name, voc12_root) for img_name in img_name_list]


def load_image_label_list_from_npy(img_name_list, label_file='data/voc12/cls_labels.npy'):
    cls_labels_dict = np.load(label_file, allow_pickle=True).item()
    return np.array([cls_labels_dict[img_name] for img_name in img_name_list])


def get_img_path(img_name, voc12_root):
    if not isinstance(img_name, str):
        img_name = decode_int_filename(img_name)
    return os.path.join(voc12_root, IMG_FOLDER_NAME, img_name + '.jpg')


def get_segmentation_path(img_name, voc12_root):
    if not isinstance(img_name, str):
        img_name = decode_int_filename(img_name)
    return os.path.join(voc12_root, SEGMENTATION_FOLDER_NAME, img_name + '.png')


def load_img_name_list(dataset_path):
    img_name_list = np.loadtxt(dataset_path, dtype=np.int32)

    return img_name_list


class Voc12ImageDataset(Dataset):
    CLASSES = CAT_LIST

    def __init__(self, img_name_list_path, voc12_root):
        self.img_name_list = load_img_name_list(img_name_list_path)
        self.voc12_root = voc12_root

        self._init_palette()

    def __len__(self):
        return len(self.img_name_list)

    def _init_palette(self):
        self.cmap = visualize.colormap()
        self.palette = ImagePalette.ImagePalette()
        for rgb in self.cmap:
            self.palette.getcolor(rgb)

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)
        img_path = get_img_path(name_str, self.voc12_root)
        img = Image.open(img_path).convert('RGB')
        return {'name': name_str, 'img': img}


class Voc12ClassificationDataset(Voc12ImageDataset):
    def __init__(self, img_name_list_path, label_file_path, voc12_root,
                 crop_size=448, scale=(0.5, 1.0), transforms=None):
        super().__init__(img_name_list_path, voc12_root)
        self.label_list = load_image_label_list_from_npy(self.img_name_list, label_file_path)

        self.transforms = transforms or tf.Compose([tf.RandomResizedCrop(crop_size, scale=scale),
                                                    tf.RandomHorizontalFlip(),
                                                    tf.RandomApply(
                                                        [tf.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3,
                                                                        hue=0.1)],
                                                        p=1.0),
                                                    tf.ToTensor(),
                                                    tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                                    ])

    def __getitem__(self, idx):
        out = super().__getitem__(idx)
        out['img'] = self.transforms(out['img'])
        out['label'] = torch.from_numpy(self.label_list[idx])
        return out


class Voc12SegmentationDatasetNoAug(Voc12ClassificationDataset):

    def __init__(self, img_name_list_path, label_file_path, voc12_root, resize_to=None):
        t = tf.Compose([tf.ToTensor(),
                        tf.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                        ])

        super().__init__(img_name_list_path, label_file_path, voc12_root, transforms=t)
        self.resize_to = resize_to

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)

        img_path = get_img_path(name_str, self.voc12_root)
        img = Image.open(img_path).convert('RGB')
        ori_size = (img.size[1], img.size[0])

        #  get ground-truth masks for visualization
        mask_path = get_segmentation_path(name_str, self.voc12_root)
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('P')
        else:
            raise ValueError(f'Mask {name_str} not found.')
            # mask = np.zeros((img.size[0], img.size[1]), dtype=np.uint8)

        if self.resize_to is not None:
            img = ttf.resize(img, size=self.resize_to, interpolation=Image.BICUBIC)
            mask = ttf.resize(mask, size=self.resize_to, interpolation=Image.NEAREST)

        mask = torch.from_numpy(np.array(mask))

        img = self.transforms(img)

        out = {"name": name_str, "img": img, "size": ori_size,
               "label": torch.from_numpy(self.label_list[idx]), "mask": mask}
        return out


class Voc12SegmentationDatasetMSF(Voc12ClassificationDataset):

    def __init__(self, img_name_list_path, label_file_path, voc12_root, resize_to=None,
                 scales=(1.0,)):
        self.scales = scales

        t = tf.Compose([tf.ToTensor(),
                        tf.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                        ])

        super().__init__(img_name_list_path, label_file_path, voc12_root, transforms=t)
        self.scales = scales
        self.resize_to = resize_to

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)

        img_path = get_img_path(name_str, self.voc12_root)
        img = Image.open(img_path).convert('RGB')
        ori_size = (img.size[1], img.size[0])   # PIL size is given as a 2-tuple (width, height).

        #  get ground-truth masks for visualization
        mask_path = get_segmentation_path(name_str, self.voc12_root)
        if os.path.exists(mask_path):
            mask = Image.open(mask_path).convert('P')
        else:
            mask = np.zeros(ori_size, dtype=np.uint8)

        if self.resize_to is not None:
            img = ttf.resize(img, size=self.resize_to, interpolation=Image.BICUBIC)
            mask = ttf.resize(mask, size=self.resize_to, interpolation=Image.NEAREST)

        mask = torch.from_numpy(np.array(mask))

        ms_img_list = []
        for s in self.scales:
            if s == 1:
                s_img = img
            else:
                height, width = img.size
                target_size = [int(np.round(height * s)), int(np.round(width * s))]
                s_img = ttf.resize(img, target_size, interpolation=Image.BICUBIC)
            s_img = self.transforms(s_img)
            ms_img_list.append(torch.stack([s_img, s_img.flip(-1)], dim=0))

        out = {"name": name_str, "img": ms_img_list, "size": ori_size,
               "label": torch.from_numpy(self.label_list[idx]), "mask": mask}
        return out


class Voc12DatasetTest(Voc12ImageDataset):

    def __init__(self, img_name_list_path, voc12_root,
                 scales=(1.0,)):
        self.scales = scales

        super().__init__(img_name_list_path, voc12_root)
        self.scales = scales
        self.transforms = tf.Compose([tf.ToTensor(),
                                      tf.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225]),
                                      ])

    def __getitem__(self, idx):
        name = self.img_name_list[idx]
        name_str = decode_int_filename(name)
        img_path = get_img_path(name_str, self.voc12_root)
        img = Image.open(img_path).convert('RGB')
        ori_size = (img.size[1], img.size[0])   # PIL size is given as a 2-tuple (width, height).

        ms_img_list = []
        for s in self.scales:
            if s == 1:
                s_img = img
            else:
                height, width = img.size
                target_size = [int(np.round(height * s)), int(np.round(width * s))]
                s_img = ttf.resize(img, target_size, interpolation=Image.BICUBIC)
            s_img = self.transforms(s_img)
            ms_img_list.append(torch.stack([s_img, s_img.flip(-1)], dim=0))
        # placeholders for ground-truth labels
        out = {"name": name_str, "img": ms_img_list, "size": ori_size,
               "label": torch.zeros(20, dtype=torch.int),
               "mask": torch.zeros(img.size[0], img.size[1], dtype=torch.int)}
        return out
