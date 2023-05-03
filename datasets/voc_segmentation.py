import os
from torch.utils.data import Dataset
from PIL import Image, ImagePalette
from datasets.utils import colormap
import datasets.transforms as tf
import logging
import datasets.augmentation as aug


class PascalVOC(Dataset):
    CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'potted-plant', 'sheep', 'sofa', 'train',
        'tv/monitor', 'ambiguous'
    ]

    CLASS_IDX = {
        'background': 0,
        'aeroplane': 1,
        'bicycle': 2,
        'bird': 3,
        'boat': 4,
        'bottle': 5,
        'bus': 6,
        'car': 7,
        'cat': 8,
        'chair': 9,
        'cow': 10,
        'diningtable': 11,
        'dog': 12,
        'horse': 13,
        'motorbike': 14,
        'person': 15,
        'potted-plant': 16,
        'sheep': 17,
        'sofa': 18,
        'train': 19,
        'tv/monitor': 20,
        'ambiguous': 255
    }
    num_classes = 21

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    PALETTE = None

    def __init__(self):
        super().__init__()
        self.PALETTE = self.get_palette()

    def get_palette(self):
        cmap = colormap()
        palette = ImagePalette.ImagePalette()
        for rgb in cmap:
            palette.getcolor(rgb)
        return palette


class VOCSegmentation(PascalVOC):
    def __init__(self, split='train.txt', root='./data', transform=None):
        super(VOCSegmentation, self).__init__()
        self.root = root
        self.split = split

        logger = logging.getLogger("global")
        assert os.path.isfile(split)
        # parse data list
        with open(split, "r") as lines:
            self.samples = [
                (
                    os.path.join(root, "JPEGImages/{}.jpg".format(line.strip())),
                    os.path.join(root, "SegmentationClassAug/{}.png".format(line.strip())),
                )
                for line in lines
            ]
        logger.info("# samples: {}".format(len(self)))

        t = tf.Compose([tf.MaskNormalise(self.MEAN, self.STD),
                        tf.MaskToTensor()])
        self.transform = transform or t

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image = Image.open(self.samples[index][0]).convert('RGB')
        mask = Image.open(self.samples[index][1]).convert('L')

        image, mask = self.transform(image, mask)

        return image[0], mask[0, 0].long()


def voc_trainset(**kwargs):
    t = aug.Compose([
        aug.ToTensor(),
        aug.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
        aug.RandResize([0.5, 2.0]),
        aug.RandomHorizontalFlip(),
        aug.Crop((513, 513), crop_type='rand', ignore_label=255)
    ])
    kwargs.update({'transform': t})
    dataset = VOCSegmentation(**kwargs)
    return dataset


def voc_valset(**kwargs):
    t = aug.Compose([
        aug.ToTensor(),
        aug.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]),
        aug.Crop((513, 513), crop_type='center', ignore_label=255)
    ])
    kwargs.update({'transform': t})
    dataset = VOCSegmentation(**kwargs)
    return dataset


if __name__ == '__main__':
    from utils import build_module
    d = build_module('datasets.voc_segmentation.voc_trainset', root='/home/pjw/VOCdevkit/VOC2012',
                     split='/home/pjw/HSS/data/voc12/train.txt')
    print(d[0])