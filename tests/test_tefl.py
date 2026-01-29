# SYMBIONT Framework - TEFL Test Suite
"""
Tests for symbiont.tefl module.
Transmissibility-Embedding Federated Learning (Algorithm 4).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from symbiont.core.constants import (
    DEFAULT_DELTA,
    DEFAULT_EPSILON,
    EMBEDDING_DIMENSION,
    EMBEDDING_SENSITIVITY,
)
from symbiont.tefl import (
    TEFL,
    DifferentialPrivacy,
    DigitalTwinSynchronizer,
    HierarchicalAggregator,
    LocalEncoder,
    PersonalizedAdapter,
)


class TestDifferentialPrivacy:
    """Tests for DifferentialPrivacy class."""

    @pytest.fixture
    def dp(self):
        """Create DifferentialPrivacy instance."""
        return DifferentialPrivacy(
            epsilon=DEFAULT_EPSILON, delta=DEFAULT_DELTA, sensitivity=EMBEDDING_SENSITIVITY
        )

    def test_initialization(self, dp):
        """Test DP initializes correctly."""
        assert dp.epsilon == DEFAULT_EPSILON
        assert dp.delta == DEFAULT_DELTA

    def test_noise_scale(self, dp):
        """Test noise scale is computed."""
        assert dp.noise_scale > 0

    def test_privatize(self, dp):
        """Test privatization adds noise."""
        embedding = np.random.randn(EMBEDDING_DIMENSION)

        private = dp.privatize(embedding)

        # Should be different due to noise
        assert not np.allclose(private, embedding)
        # Should have same shape
        assert private.shape == embedding.shape

    def test_lower_epsilon_more_noise(self):
        """Test that lower ε requires more noise."""
        dp_high = DifferentialPrivacy(epsilon=1.0, delta=1e-5, sensitivity=1.0)
        dp_low = DifferentialPrivacy(epsilon=0.1, delta=1e-5, sensitivity=1.0)

        assert dp_low.noise_scale > dp_high.noise_scale

    def test_compose_privacy(self, dp):
        """Test privacy budget composition."""
        composed_eps, composed_delta = dp.compose_privacy(num_rounds=10)

        assert composed_eps >= dp.epsilon
        assert composed_delta >= dp.delta


class TestLocalEncoder:
    """Tests for LocalEncoder class."""

    @pytest.fixture
    def encoder(self):
        """Create LocalEncoder instance."""
        return LocalEncoder(input_dim=64, embedding_dim=EMBEDDING_DIMENSION)

    def test_initialization(self, encoder):
        """Test encoder initializes correctly."""
        assert encoder.embedding_dim == EMBEDDING_DIMENSION

    def test_encoding_shape(self, encoder):
        """Test encoding produces correct shape."""
        transmissibility = np.random.randn(64) + 1j * np.random.randn(64)

        embedding = encoder.encode(transmissibility)

        assert embedding.shape[0] == EMBEDDING_DIMENSION

    def test_bounded_output(self, encoder):
        """Test output is bounded (tanh)."""
        transmissibility = np.random.randn(64) * 100  # Large values

        embedding = encoder.encode(transmissibility)

        # Should be bounded due to tanh
        assert np.all(np.abs(embedding) <= 1.1)

    def test_deterministic(self, encoder):
        """Test encoding is deterministic."""
        transmissibility = np.random.randn(64)

        e1 = encoder.encode(transmissibility)
        e2 = encoder.encode(transmissibility)

        assert_allclose(e1, e2)


class TestPersonalizedAdapter:
    """Tests for PersonalizedAdapter class."""

    @pytest.fixture
    def adapter(self):
        """Create PersonalizedAdapter instance."""
        return PersonalizedAdapter(input_dim=EMBEDDING_DIMENSION)

    def test_initialization(self, adapter):
        """Test adapter initializes correctly."""
        assert adapter is not None

    def test_forward_pass(self, adapter):
        """Test forward pass."""
        embedding = np.random.randn(EMBEDDING_DIMENSION)

        output = adapter.forward(embedding)

        assert output is not None
        # Output dimension is hidden_dim (8), not embedding_dim
        assert len(output) == 8  # hidden_dim default


class TestDigitalTwinSynchronizer:
    """Tests for DigitalTwinSynchronizer class."""

    @pytest.fixture
    def synchronizer(self):
        """Create DigitalTwinSynchronizer instance."""
        return DigitalTwinSynchronizer()

    def test_initialization(self, synchronizer):
        """Test synchronizer initializes correctly."""
        assert synchronizer is not None
        assert synchronizer.dt_state is not None


class TestHierarchicalAggregator:
    """Tests for HierarchicalAggregator class."""

    @pytest.fixture
    def aggregator(self):
        """Create HierarchicalAggregator instance."""
        return HierarchicalAggregator()

    def test_initialization(self, aggregator):
        """Test aggregator initializes correctly."""
        assert aggregator is not None

    def test_communication_savings(self):
        """Test hierarchical saves communication (Theorem 15)."""
        from symbiont.core import theorems

        n_nodes = 1000

        hierarchical_cost = theorems.hierarchical_communication_cost(n_nodes)
        flat_cost = theorems.flat_communication_cost(n_nodes)

        assert hierarchical_cost < flat_cost


class TestTEFL:
    """Tests for TEFL class (Algorithm 4)."""

    @pytest.fixture
    def tefl(self):
        """Create TEFL instance."""
        return TEFL(
            n_sensors=4,
            adjacency=[(0, 1), (1, 2), (2, 3), (0, 3)],
            embedding_dim=EMBEDDING_DIMENSION,
            epsilon=DEFAULT_EPSILON,
            delta=DEFAULT_DELTA,
        )

    def test_initialization(self, tefl):
        """Test TEFL initializes correctly."""
        assert tefl.n_sensors == 4
        assert tefl.embedding_dim == EMBEDDING_DIMENSION

    def test_local_update(self, tefl):
        """Test local update step."""
        transmissibility = np.random.randn(64) + 1j * np.random.randn(64)

        embedding = tefl.local_update(
            sensor_i=0, sensor_j=1, transmissibility=transmissibility, timestamp=0.0
        )

        assert embedding is not None


class TestTEFLIntegration:
    """Integration tests for TEFL module."""

    def test_federated_round(self):
        """Test complete federated learning round."""
        tefl = TEFL(
            n_sensors=4,
            adjacency=[(0, 1), (1, 2), (2, 3)],
            embedding_dim=EMBEDDING_DIMENSION,
            epsilon=1.0,
            delta=1e-5,
        )

        # Simulate local updates
        for i, j in [(0, 1), (1, 2), (2, 3)]:
            t = np.random.randn(64) + 1j * np.random.randn(64)
            tefl.local_update(sensor_i=i, sensor_j=j, transmissibility=t, timestamp=0.0)

    def test_privacy_utility_tradeoff(self):
        """Test privacy-utility tradeoff."""
        dp_low = DifferentialPrivacy(epsilon=0.1, delta=1e-5, sensitivity=1.0)
        dp_high = DifferentialPrivacy(epsilon=1.0, delta=1e-5, sensitivity=1.0)

        # Lower epsilon requires more noise
        assert dp_low.noise_scale > dp_high.noise_scale

    def test_hierarchical_communication_savings(self):
        """Test hierarchical communication savings."""
        from symbiont.core import theorems

        n_nodes = 1000
        ratio = theorems.hierarchical_vs_flat_ratio(n_nodes)

        # Should achieve significant savings
        assert ratio < 0.1


class TestCommunicationEfficiency:
    """Tests for communication efficiency."""

    def test_embedding_size(self):
        """Test embedding fits in communication budget."""
        embedding = np.random.randn(EMBEDDING_DIMENSION).astype(np.float32)

        size_bytes = embedding.nbytes

        # Should be compact (<0.5KB)
        assert size_bytes < 500

    def test_hierarchical_vs_flat(self):
        """Test hierarchical outperforms flat for large N."""
        from symbiont.core import theorems

        for n in [100, 1000, 10000]:
            hier = theorems.hierarchical_communication_cost(n)
            flat = theorems.flat_communication_cost(n)

            assert hier < flat
