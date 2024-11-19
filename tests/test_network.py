import pytest

import torch
from torch.utils.data import DataLoader, random_split
from provided.network import SimpleNeuralNetwork
from provided.loader import MultiProcessDataset

from provided.constants import DATA_DIR

from provided.network import SimpleNeuralNetwork

@pytest.fixture(scope="function")
def data():
    dataset = MultiProcessDataset(DATA_DIR / "3d_data.csv")
    dataloader = DataLoader(
        dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=True
    )

    ###########################################################################
    # Split data into training and test sets
    ###########################################################################

    torch.manual_seed(42)  # Set the seed for reproducibility
    training_size = int(0.8 * len(dataset))  # 80% of data for training
    test_size = len(dataset) - training_size  # 20% of data for testing
    train_dataset, test_dataset = random_split(dataset, [training_size, test_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=32, shuffle=True, num_workers=2, drop_last=True
    )

    return {"train_dataloader": train_dataloader, "test_dataloader": test_dataloader}


class TestSimpleNeuralNetwork:

    @pytest.mark.parametrize(
        "input_dim, hidden_sizes, output_size",
        [
            (3, [4, 4], 1),
            (3, [4, 2], 1),
            (3, [2, 4], 1),
        ],
    )
    def test_initialization(
        self, input_dim: int, hidden_sizes: list[int], output_size: int
    ):
        """
        Test SimpleNeuralNetwork initialization.
        """
        network = SimpleNeuralNetwork(
            input_dim=input_dim, hidden_sizes=hidden_sizes, output_size=output_size
        )

        print(network.W1.shape)
        print(network.W2.shape)
        print(network.W3.shape)
        assert network.W1.shape == (hidden_sizes[0], input_dim), f"W1 should have shape ({hidden_sizes[0]}, {input_dim})"
        assert network.W2.shape == (hidden_sizes[1], hidden_sizes[0]), f"W2 should have shape ({hidden_sizes[1]}, {hidden_sizes[0]})"
        assert network.W3.shape == (output_size, hidden_sizes[1]), f"W3 should have shape ({output_size}, {hidden_sizes[1]})"

    @pytest.mark.parametrize(
        "input_dim, hidden_sizes, output_size",
        [
            (3, [4, 4], 1),
            (3, [4, 2], 1),
            (3, [2, 4], 1),
        ],
    )
    def test_forward(self, input_dim: int, hidden_sizes: list[int], output_size: int, data):
        """
        Test SimpleNeuralNetwork forward pass.
        """
        network = SimpleNeuralNetwork(
            input_dim=input_dim, hidden_sizes=hidden_sizes, output_size=output_size
        )

        # Unpack the first batch (X and labels)
        first_batch = next(iter(data["test_dataloader"]))
        X_batch, labels_batch = first_batch

        # Reshape to (batch_size, 1)
        X = X_batch.float()

        # Forward pass
        output = network.forward(X)

        assert output.shape == (32, 1), f"Output shape should be (32, 1)"

    @pytest.mark.parametrize(
        "input_dim, hidden_sizes, output_size",
        [
            (3, [4, 4], 1),
            (3, [4, 2], 1),
            (3, [2, 4], 1),
        ],
    )
    def test_backward(self, input_dim: int, hidden_sizes: list[int], output_size: int, data):
        """
        Test SimpleNeuralNetwork backward pass.
        """
        network = SimpleNeuralNetwork(
            input_dim=input_dim, hidden_sizes=hidden_sizes, output_size=output_size
        )

        # Unpack the first batch (X and labels)
        first_batch = next(iter(data["test_dataloader"]))
        X_batch, labels_batch = first_batch

        # Reshape to (batch_size, 1)
        X = X_batch.float()
        Y = labels_batch.float().reshape(-1, 1)  

        # Forward pass
        output = network.forward(X)

        # Compute loss (Mean Squared Error)
        loss_1 = torch.mean((output - Y) ** 2)
        print("Loss 1: ", loss_1)
        assert loss_1 is not None, "Loss should not be None"

        # Backward pass
        network.backward(Y, learning_rate=0.0001)


        # Second forward pass
        second_batch = next(iter(data["test_dataloader"]))
        X_batch, labels_batch = second_batch
        X = X_batch.float()
        Y = labels_batch.float().reshape(-1, 1)

        output = network.forward(X)
        loss_2 = torch.mean((output - Y) ** 2)
        print("Loss 2: ", loss_2)
        assert loss_2 is not None, "Loss should not be None"


        assert network.W1 is not None, "W1 should not be None"
        assert network.W2 is not None, "W2 should not be None"
        assert network.W3 is not None, "W3 should not be None"
        assert network.b1 is not None, "b1 should not be None"
        assert network.b2 is not None, "b2 should not be None"
        assert network.b3 is not None, "b3 should not be None"
