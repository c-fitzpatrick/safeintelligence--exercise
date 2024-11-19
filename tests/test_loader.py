import torch

import pytest
import pathlib

from provided.loader import SingleProcessDataset, MultiProcessDataset
from provided.constants import DATA_DIR


class TestLoader:

    @pytest.mark.parametrize(
        "csv_file, n",
        [
            (DATA_DIR / "3d_data.csv", 10000),
            (DATA_DIR / "3d_data_10.csv", 10),
        ],
    )
    def test_single_loader(self, csv_file: pathlib.Path, n: int):
        """
        Test SingleProcessDataset. Confirm that features and labels are of the correct type and shape.
        """
        dataset = SingleProcessDataset(csv_file)

        assert (
            dataset.features.dtype == torch.float32
        ), "Features should be of type torch.float32"
        assert (
            dataset.labels.dtype == torch.int64
        ), "Labels should be of type torch.int64"

        assert dataset.features.shape == (n, 3), f"Features should have shape ({n}, 3)"
        assert dataset.labels.shape == (n,), f"Labels should have shape ({n},)"

    @pytest.mark.parametrize(
        "csv_file, n, chunksize",
        [
            (DATA_DIR / "3d_data.csv", 10000, 1000),
            (DATA_DIR / "3d_data.csv", 10000, 5000),
            (DATA_DIR / "3d_data_10.csv", 10, 1),
            (DATA_DIR / "3d_data_10.csv", 10, 5),
        ],
    )
    def test_multi_loader(self, csv_file: pathlib.Path, n: int, chunksize: int):
        """
        Test MultiProcessDataset. Confirm that features and labels are of the correct type and shape.
        """
        dataset = MultiProcessDataset(csv_file=csv_file, chunksize=chunksize)

        assert (
            dataset.features.dtype == torch.float32
        ), "Features should be of type torch.float32"
        assert (
            dataset.labels.dtype == torch.int64
        ), "Labels should be of type torch.int64"

        assert dataset.features.shape == (n, 3), f"Features should have shape ({n}, 3)"
        assert dataset.labels.shape == (n,), f"Labels should have shape ({n},)"
