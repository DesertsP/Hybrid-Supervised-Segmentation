import os
from torch.utils.data import Dataset
from PIL import Image, ImagePalette
from datasets.utils import colormap
import datasets.transforms as tf
import logging
import datasets.augmentation as aug


class Cityscapes(Dataset):
    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')
    NUM_CLASS = 19

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

    @property
    def num_classes(self):
        return self.NUM_CLASS


class CityscapesSegmentation(Cityscapes):
    def __init__(self, split='train.txt', root='./data', is_unlabeled=False,
                 transform=None, ignore_label=255):
        super(CityscapesSegmentation, self).__init__()
        self.root = root
        self.split = split
        self.is_unlabeled = is_unlabeled
        self.ignore_label = ignore_label

        logger = logging.getLogger("global")
        assert os.path.isfile(split)
        # parse data list
        with open(split, "r") as lines:
            self.samples = [
                (
                    os.path.join(root, line.strip()),
                    os.path.join(root, 'gtFine', line.strip()[12:-15] + "gtFine_labelTrainIds.png"),
                )
                for line in lines
            ]
        logger.info("# samples: {}".format(len(self)))

        t = aug.ToTensor()
        self.transform = transform or t

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image = Image.open(self.samples[index][0]).convert('RGB')
        mask = Image.open(self.samples[index][1])  # .convert('L')

        image, mask = self.transform(image, mask)
        image /= 255.0
        results = dict(image=image[0], index=index)
        if not self.is_unlabeled:
            # target only given for labeled data
            mask = mask[0, 0].long()
            mask[mask < 0] = self.ignore_label
            mask[mask >= self.NUM_CLASS] = self.ignore_label
            results.update(dict(target=mask))

        return results


def cityscapes_trainset(**kwargs):
    t = aug.Compose([
        aug.ToTensor(),
        aug.RandResize([0.5, 2.0]),
        aug.RandomHorizontalFlip(),
        aug.Crop((769, 769), crop_type='rand', ignore_label=255)
    ])
    kwargs.update({'transform': t})
    dataset = CityscapesSegmentation(**kwargs)
    return dataset


def cityscapes_valset(**kwargs):
    t = aug.Compose([
        aug.ToTensor(),
        aug.Crop((769, 769), crop_type='center', ignore_label=255)
    ])
    kwargs.update({'transform': t})
    dataset = CityscapesSegmentation(**kwargs)
    return dataset


def cityscapes_testset(**kwargs):
    t = aug.ToTensor()
    kwargs.update({'transform': t})
    dataset = CityscapesSegmentation(**kwargs)
    return dataset


if __name__ == '__main__':
    from utils import build_module

    d = build_module('datasets.voc_segmentation.voc_trainset', root='/home/pjw/VOCdevkit/VOC2012',
                     split='/home/pjw/HSS/data/voc12/train.txt')
    print(d[0])
