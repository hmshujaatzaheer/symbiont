# SYMBIONT Framework - SMIS Test Suite
"""
Tests for symbiont.smis module.
Seismic-Mode Inspired Synchronization (Algorithm 1).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from symbiont.core.constants import (
    DEFAULT_SAMPLE_RATE,
    STEEL_LONGITUDINAL_VELOCITY,
    STEEL_TRANSVERSE_VELOCITY,
)
from symbiont.smis import (
    SMIS,
    MultiModalWaveDecomposer,
    VelocityFieldEstimator,
)


class TestMultiModalWaveDecomposer:
    """Tests for MultiModalWaveDecomposer class."""

    @pytest.fixture
    def decomposer(self):
        """Create MultiModalWaveDecomposer instance."""
        return MultiModalWaveDecomposer(
            sample_rate=DEFAULT_SAMPLE_RATE,
            c_L=STEEL_LONGITUDINAL_VELOCITY,
            c_T=STEEL_TRANSVERSE_VELOCITY,
        )

    def test_initialization(self, decomposer):
        """Test decomposer initializes correctly."""
        assert decomposer is not None
        assert decomposer.c_L == STEEL_LONGITUDINAL_VELOCITY
        assert decomposer.c_T == STEEL_TRANSVERSE_VELOCITY

    def test_compute_modal_separation(self, decomposer):
        """Test modal separation computation."""
        distance = 10.0  # meters

        separation = decomposer.compute_modal_separation(distance)

        # Should be positive time difference
        assert separation > 0

    def test_extract_modal_arrivals(self, decomposer):
        """Test modal arrival extraction."""
        # Create synthetic signal with two arrivals
        t = np.linspace(0, 0.1, 1000)
        signal = np.zeros_like(t)

        # L-wave arrival at 0.01s
        signal[int(0.01 * 10000) : int(0.015 * 10000)] = np.sin(np.linspace(0, 4 * np.pi, 50))
        # S-wave arrival at 0.02s
        signal[int(0.02 * 10000) : int(0.025 * 10000)] = np.sin(np.linspace(0, 4 * np.pi, 50))

        arrivals = decomposer.extract_modal_arrivals(signal, threshold=0.3)

        # Returns Tuple[float, float] - (t_L, t_T)
        assert isinstance(arrivals, tuple)
        assert len(arrivals) == 2
        t_L, t_T = arrivals
        assert isinstance(t_L, float)
        assert isinstance(t_T, float)


class TestVelocityFieldEstimator:
    """Tests for VelocityFieldEstimator class."""

    @pytest.fixture
    def estimator(self):
        """Create VelocityFieldEstimator instance."""
        # Use 2D positions in column form to avoid atleast_2d issues
        positions = np.array([[0], [2.5], [5], [7.5], [10.0]])  # (5, 1) - 1D positions
        return VelocityFieldEstimator(
            sensor_positions=positions, initial_velocity=STEEL_LONGITUDINAL_VELOCITY
        )

    def test_initialization(self, estimator):
        """Test estimator initializes correctly."""
        assert estimator is not None

    def test_compute_velocity_gradient(self, estimator):
        """Test velocity gradient computation."""
        gradient = estimator.compute_velocity_gradient()

        # Should return an array (could be [0.0] if not enough points)
        assert isinstance(gradient, np.ndarray)
        assert len(gradient) >= 1


class TestSMIS:
    """Tests for SMIS class (Algorithm 1)."""

    @pytest.fixture
    def smis(self):
        """Create SMIS instance."""
        positions = np.linspace(0, 10, 4)
        return SMIS(
            sensor_positions=positions,
            sample_rate=DEFAULT_SAMPLE_RATE,
            c_L=STEEL_LONGITUDINAL_VELOCITY,
            c_T=STEEL_TRANSVERSE_VELOCITY,
        )

    def test_initialization(self, smis):
        """Test SMIS initializes correctly."""
        assert smis is not None
        assert smis.n_sensors == 4

    def test_synchronize_timestamps(self, smis):
        """Test timestamp synchronization."""
        # Raw timestamps with clock offsets
        timestamps = {0: 1000.0, 1: 1000.001, 2: 999.998, 3: 1000.002}

        synced = smis.synchronize_timestamps(timestamps)

        assert len(synced) == 4


class TestSMISTheorems:
    """Test SMIS theorem compliance."""

    def test_theorem_2_sqrt_n_scaling(self):
        """Test that SE scales as 1/√N per Theorem 2."""
        from symbiont.core import theorems

        se_100 = theorems.smis_synchronization_bound(1e-6, 5000, 100)
        se_400 = theorems.smis_synchronization_bound(1e-6, 5000, 400)

        # SE(400) should be half of SE(100)
        assert_allclose(se_400, se_100 / 2, rtol=0.01)

    def test_theorem_3_multimodal_improvement(self):
        """Test multimodal improvement per Theorem 3."""
        from symbiont.core import theorems

        c_L = 5000.0
        c_T = 3000.0

        se_single = theorems.smis_synchronization_bound(1e-6, c_L, 100)
        se_multi = theorems.multimodal_synchronization_error(1e-6, c_L, c_T, 100)

        # Multimodal should be better
        assert se_multi < se_single
