import pytest

import torch
from torch.utils.data import DataLoader, random_split
from provided.network import SimpleNeuralNetwork
from provided.loader import MultiProcessDataset
from provided.bounds import IntervalBoundPropagation

from provided.constants import DATA_DIR

class TestIntervalBoundPropagation:

    @pytest.mark.parametrize(
        "input_dim, hidden_sizes, output_size",
        [
            (3, [4, 4], 1),
            (3, [4, 2], 1),
            (3, [2, 4], 1),
        ],
    )
    def test_compute_bounds_forward(
        self, input_dim: int, hidden_sizes: list[int], output_size: int
    ):
        """
        Test bound propagation.
        """
        dataset = MultiProcessDataset(DATA_DIR / '3d_data.csv')
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=True)

        ###########################################################################
        # Split data into training and test sets
        ###########################################################################

        torch.manual_seed(42) # Set the seed for reproducibility
        training_size = int(0.8 * len(dataset)) # 80% of data for training
        test_size = len(dataset) - training_size # 20% of data for testing
        train_dataset, test_dataset = random_split(dataset, [training_size, test_size])

        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=True)

        ###########################################################################
        # Train the network
        ###########################################################################

        network = SimpleNeuralNetwork(
            input_dim=input_dim,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
        )

        network.train(train_dataloader, epochs=2, batch_size=32, learning_rate=0.0001)


        # Unpack the first batch (X and labels)
        first_batch = next(iter(test_dataloader))
        X_batch, labels_batch = first_batch

        # Define an epsilon to add (upper) / subtract (lower) 
        # from the input features
        eps = 0.1

        # Define input bounds
        # Expected shape: (batch_size, input_dim, 2)
        # 2 for lower and upper bounds
        input_bounds = torch.zeros((32, 3, 2), dtype=torch.float32)
        input_bounds[..., 0] = X_batch - eps
        input_bounds[..., 1] = X_batch + eps

        # Compute bounds for the first layer
        print(f"Finding bounds for batched input: shape {input_bounds.shape}")

        bp = IntervalBoundPropagation(network)
        ouput_bounds = bp.compute_bounds_forward(input_bounds)

        assert ouput_bounds.shape == (32, output_size, 2)

        ouput_bounds = bp.compute_bounds_forward_alt(input_bounds)

        assert ouput_bounds.shape == (32, output_size, 2)
