import torch

import pytest


from provided.loader import SingleProcessDataset, MultiProcessDataset
from provided.constants import DATA_DIR


class TestLoader:

    def test_single_loader(self):
        csv_file = DATA_DIR / "3d_data.csv"
        dataset = SingleProcessDataset(csv_file)

        assert dataset.features.dtype == torch.float32, "Features should be of type torch.float32"
        assert dataset.labels.dtype == torch.int64, "Labels should be of type torch.int64"

        assert dataset.features.shape == (10000, 3), "Features should have shape (10000, 3)"
        assert dataset.labels.shape == (10000,), "Labels should have shape (10000,)"