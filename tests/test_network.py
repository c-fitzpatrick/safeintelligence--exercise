import pytest

from provided.network import SimpleNeuralNetwork

class TestSimpleNeuralNetwork:

    def test_initialization(self):
        """
        Test SimpleNeuralNetwork initialization.
        """
        network = SimpleNeuralNetwork(
            input_dim=3,
            hidden_sizes=[4, 4],
            output_size=1
        )

        assert network.W1.shape == (4, 3), "W1 should have shape (4, 3)"
        assert network.W2.shape == (4, 4), "W2 should have shape (4, 4)"
        assert network.W3.shape == (1, 4), "W3 should have shape (1, 4)"
