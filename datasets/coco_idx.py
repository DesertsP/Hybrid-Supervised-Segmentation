import os
from torch.utils.data import Dataset
from PIL import Image, ImagePalette
from datasets.utils import colormap
import datasets.transforms as tf
import logging
import datasets.augmentation as aug


class COCODataset(Dataset):
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    NUM_CLASS = 81
    CLASSES = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
               'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
               'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
               'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
               'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
               'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
               'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
               'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

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


class COCOSegmentation(COCODataset):
    def __init__(self, split='train.txt', root='./data', image_dir='train2017', label_dir='train_mask', is_unlabeled=False,
                 transform=None, ignore_label=255):
        super(COCOSegmentation, self).__init__()
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
                    os.path.join(root, "{}/{}.jpg".format(image_dir, line.strip())),
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


def coco_trainset(**kwargs):
    t = aug.Compose([
        aug.ToTensor(),
        aug.RandResize([0.5, 2.0]),
        aug.RandomHorizontalFlip(),
        aug.Crop((513, 513), crop_type='rand', ignore_label=255)
    ])
    kwargs.update({'transform': t})
    dataset = COCOSegmentation(**kwargs)
    return dataset


def coco_valset(**kwargs):
    t = aug.Compose([
        aug.ToTensor(),
        aug.Crop((513, 513), crop_type='center', ignore_label=255)
    ])
    kwargs.update({'transform': t})
    dataset = COCOSegmentation(**kwargs)
    return dataset


def coco_testset(**kwargs):
    t = aug.ToTensor()
    kwargs.update({'transform': t})
    dataset = COCOSegmentation(**kwargs)
    return dataset
