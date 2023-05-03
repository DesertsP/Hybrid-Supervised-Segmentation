# Copied from chainercv

from __future__ import division

import numpy as np
import six


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def calc_semantic_segmentation_confusion(pred_gt_pairs):
    """Collect a confusion matrix.

    The number of classes :math:`n\_class` is
    :math:`max(pred\_labels, gt\_labels) + 1`, which is
    the maximum class id of the inputs added by one.

    Args:
        pred_labels (iterable of numpy.ndarray): See the table in
            :func:`chainercv.evaluations.eval_semantic_segmentation`.
        gt_labels (iterable of numpy.ndarray): See the table in
            :func:`chainercv.evaluations.eval_semantic_segmentation`.

    Returns:
        numpy.ndarray:
        A confusion matrix. Its shape is :math:`(n\_class, n\_class)`.
        The :math:`(i, j)` th element corresponds to the number of pixels
        that are labeled as class :math:`i` by the ground truth and
        class :math:`j` by the prediction.

    """
    n_class = 0
    confusion = np.zeros((n_class, n_class), dtype=np.int64)
    for pred_label, gt_label in pred_gt_pairs:
        if pred_label.ndim != 2 or gt_label.ndim != 2:
            raise ValueError('ndim of labels should be two.')
        if pred_label.shape != gt_label.shape:
            raise ValueError('Shape of ground truth and prediction should'
                             ' be same.')
        pred_label = pred_label.flatten()
        gt_label = gt_label.flatten()

        # Dynamically expand the confusion matrix if necessary.
        lb_max = np.max((pred_label, gt_label))
        if lb_max >= n_class:
            expanded_confusion = np.zeros(
                (lb_max + 1, lb_max + 1), dtype=np.int64)
            expanded_confusion[0:n_class, 0:n_class] = confusion

            n_class = lb_max + 1
            confusion = expanded_confusion

        # Count statistics from valid pixels.
        mask = gt_label >= 0
        confusion += np.bincount(
            n_class * gt_label[mask].astype(int) +
            pred_label[mask], minlength=n_class ** 2).reshape((n_class, n_class))

    return confusion


def calc_semantic_segmentation_iou(confusion):
    """Calculate Intersection over Union with a given confusion matrix.

    The definition of Intersection over Union (IoU) is as follows,
    where :math:`N_{ij}` is the number of pixels
    that are labeled as class :math:`i` by the ground truth and
    class :math:`j` by the prediction.

    * :math:`\\text{IoU of the i-th class} =  \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`

    Args:
        confusion (numpy.ndarray): A confusion matrix. Its shape is
            :math:`(n\_class, n\_class)`.
            The :math:`(i, j)` th element corresponds to the number of pixels
            that are labeled as class :math:`i` by the ground truth and
            class :math:`j` by the prediction.

    Returns:
        numpy.ndarray:
        An array of IoUs for the :math:`n\_class` classes. Its shape is
        :math:`(n\_class,)`.

    """
    iou_denominator = (confusion.sum(axis=1) + confusion.sum(axis=0) -
                       np.diag(confusion))
    iou = np.diag(confusion) / iou_denominator
    return iou


def eval_semantic_segmentation(pred_labels, gt_labels):
    """Evaluate metrics used in Semantic Segmentation.

    This function calculates Intersection over Union (IoU), Pixel Accuracy
    and Class Accuracy for the task of semantic segmentation.

    The definition of metrics calculated by this function is as follows,
    where :math:`N_{ij}` is the number of pixels
    that are labeled as class :math:`i` by the ground truth and
    class :math:`j` by the prediction.

    * :math:`\\text{IoU of the i-th class} =  \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`
    * :math:`\\text{mIoU} = \\frac{1}{k} \
        \\sum_{i=1}^k \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij} + \\sum_{j=1}^k N_{ji} - N_{ii}}`
    * :math:`\\text{Pixel Accuracy} =  \
        \\frac \
        {\\sum_{i=1}^k N_{ii}} \
        {\\sum_{i=1}^k \\sum_{j=1}^k N_{ij}}`
    * :math:`\\text{Class Accuracy} = \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij}}`
    * :math:`\\text{Mean Class Accuracy} = \\frac{1}{k} \
        \\sum_{i=1}^k \
        \\frac{N_{ii}}{\\sum_{j=1}^k N_{ij}}`

    The more detailed description of the above metrics can be found in a
    review on semantic segmentation [#]_.

    The number of classes :math:`n\_class` is
    :math:`max(pred\_labels, gt\_labels) + 1`, which is
    the maximum class id of the inputs added by one.

    .. [#] Alberto Garcia-Garcia, Sergio Orts-Escolano, Sergiu Oprea, \
    Victor Villena-Martinez, Jose Garcia-Rodriguez. \
    `A Review on Deep Learning Techniques Applied to Semantic Segmentation \
    <https://arxiv.org/abs/1704.06857>`_. arXiv 2017.

    Args:
        pred_labels (iterable of numpy.ndarray): See the table below.
        gt_labels (iterable of numpy.ndarray): See the table below.

    .. csv-table::
        :header: name, shape, dtype, format

        :obj:`pred_labels`, ":math:`[(H, W)]`", :obj:`int32`, \
        ":math:`[0, \#class - 1]`"
        :obj:`gt_labels`, ":math:`[(H, W)]`", :obj:`int32`, \
        ":math:`[-1, \#class - 1]`"

    Returns:
        dict:

        The keys, value-types and the description of the values are listed
        below.

        * **iou** (*numpy.ndarray*): An array of IoUs for the \
            :math:`n\_class` classes. Its shape is :math:`(n\_class,)`.
        * **miou** (*float*): The average of IoUs over classes.
        * **pixel_accuracy** (*float*): The computed pixel accuracy.
        * **class_accuracy** (*numpy.ndarray*): An array of class accuracies \
            for the :math:`n\_class` classes. \
            Its shape is :math:`(n\_class,)`.
        * **mean_class_accuracy** (*float*): The average of class accuracies.

    """
    # Evaluation code is based on
    # https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/
    # score.py#L37
    confusion = calc_semantic_segmentation_confusion(
        pred_labels, gt_labels)
    iou = calc_semantic_segmentation_iou(confusion)
    pixel_accuracy = np.diag(confusion).sum() / confusion.sum()
    class_accuracy = np.diag(confusion) / np.sum(confusion, axis=1)

    return {'iou': iou, 'miou': np.nanmean(iou),
            'pixel_accuracy': pixel_accuracy,
            'class_accuracy': class_accuracy,
            'mean_class_accuracy': np.nanmean(class_accuracy)}


def eval_custom(confusion, class_names=None, use_dice=False):
    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj + 1e-8
    # fp = 1. - gtj / denominator # TP+FN / (TP+FN+FP)
    # fn = 1. - resj / denominator
    fn = 1 - gtjresj / gtj  # recall TP / (TP+FN)
    fp = 1 - gtjresj / resj  # precision
    iou = gtjresj / denominator
    dice = 2. * gtjresj / (gtj + resj)

    # print(f'fp for BG: {fp[0]:.4f}, fn for BG: {fn[0]:.4f}')
    print(f'mean fp for FG: {np.nanmean(fp[1:]):.4f}, mean fn for FG: {np.nanmean(fn[1:]):.4f}')
    print(f'mDice: {np.nanmean(dice[1:])}' if use_dice else f'mIoU: {np.nanmean(iou)}')
    if class_names:
        print(
            f'{"class": <15}\t {"dice":<6}\t {"fp":<6}\t {"fn":<6}' if use_dice else f'{"class": <15}\t {"iou":<6}\t {"fp":<6}\t {"fn":<6}')
        for name, i, j, k in zip(class_names, dice if use_dice else iou, fp, fn):
            print(f'{name: <15}\t {i:.4f}\t {j:.4f}\t {k:.4f}')
    print()
    return {'miou': np.nanmean(iou[1:]),
            'mdice': np.nanmean(dice[1:]),
            'mfp': np.nanmean(fp[1:]),
            'mfn': np.nanmean(fn[1:])}
