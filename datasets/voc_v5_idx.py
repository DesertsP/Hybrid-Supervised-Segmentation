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
    NUM_CLASS = 21

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


class VOCSegmentation(PascalVOC):
    def __init__(self, split='train.txt', root='./data', label_dir='SegmentationClassAug', is_unlabeled=False,
                 transform=None, ignore_label=255):
        super(VOCSegmentation, self).__init__()
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
                    os.path.join(root, "JPEGImages/{}.jpg".format(line.strip())),
                    os.path.join(root, "{}/{}.png".format(label_dir, line.strip())),
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
        mask = Image.open(self.samples[index][1])   # .convert('L')

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


def voc_trainset(**kwargs):
    t = aug.Compose([
        aug.ToTensor(),
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
        aug.Crop((513, 513), crop_type='center', ignore_label=255)
    ])
    kwargs.update({'transform': t})
    dataset = VOCSegmentation(**kwargs)
    return dataset


def voc_testset(**kwargs):
    t = aug.ToTensor()
    kwargs.update({'transform': t})
    dataset = VOCSegmentation(**kwargs)
    return dataset
