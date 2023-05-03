from torch.utils.data import Subset, random_split, Dataset
from typing import Sequence, Optional, Generator, List
import torch


class OverSampledSubset(Dataset):
    r"""
    Subset of a dataset at specified indices.

    Args:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset: Dataset, indices: Sequence[int], length=None) -> None:
        self.dataset = dataset
        self.indices = indices
        self.length = length

    def __getitem__(self, idx):
        idx = idx % len(self.indices)
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return self.length if self.length else len(self.indices)


