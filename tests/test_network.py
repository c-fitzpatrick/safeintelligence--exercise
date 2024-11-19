import pytest

from provided.network import SimpleNeuralNetwork

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
