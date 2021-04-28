"""
__init__.py.py - A PostHocExplainerEvaluation file
Copyright (C) 2021  Zach Carmichael
"""
from posthoceval.datasets.tiny_mnist import TinyMNISTDataset
from posthoceval.datasets.compas import COMPASDataset
from posthoceval.datasets.boston import BostonDataset
from posthoceval.datasets.dataset import Dataset
from posthoceval.datasets.dataset import CustomDataset

__all__ = [
    'Dataset', 'CustomDataset',
    'COMPASDataset', 'BostonDataset', 'TinyMNISTDataset',
]
