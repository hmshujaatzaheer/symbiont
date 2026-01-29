# SYMBIONT Framework - ECCP Test Suite
"""
Tests for symbiont.eccp module.
Energy-Correlated Coordination Protocol (Algorithm 3).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from symbiont.eccp import (
    ECCP,
    ECCPState,
    EnergyCorrelationAnalyzer,
    EnergyHarvester,
    NashEquilibriumScheduler,
    PredictiveMDPScheduler,
)


class TestEnergyHarvester:
    """Tests for EnergyHarvester class."""

    @pytest.fixture
    def harvester(self):
        """Create EnergyHarvester instance."""
        return EnergyHarvester()

    def test_initialization(self, harvester):
        """Test harvester initializes correctly."""
        assert harvester is not None

    def test_compute_power_positive(self, harvester):
        """Test power is positive for valid inputs."""
        power = harvester.compute_power(amplitude=1e-6, frequency=100)
        assert power > 0

    def test_power_scales_with_frequency_squared(self, harvester):
        """Test P ∝ f²."""
        p1 = harvester.compute_power(amplitude=1e-6, frequency=100)
        p2 = harvester.compute_power(amplitude=1e-6, frequency=200)

        assert_allclose(p2 / p1, 4.0, rtol=0.1)

    def test_power_scales_with_amplitude_squared(self, harvester):
        """Test P ∝ A²."""
        p1 = harvester.compute_power(amplitude=1e-6, frequency=100)
        p2 = harvester.compute_power(amplitude=2e-6, frequency=100)

        assert_allclose(p2 / p1, 4.0, rtol=0.1)

    def test_compute_energy_from_signal(self, harvester):
        """Test energy computation from signal."""
        t = np.linspace(0, 0.1, 1000)
        signal = 1e-6 * np.sin(2 * np.pi * 100 * t)

        energy = harvester.compute_energy_from_signal(signal, sample_rate=10000)
        assert energy > 0

    def test_zero_amplitude_zero_power(self, harvester):
        """Test zero amplitude yields zero power."""
        power = harvester.compute_power(amplitude=0, frequency=100)
        assert power == 0


class TestEnergyCorrelationAnalyzer:
    """Tests for EnergyCorrelationAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create EnergyCorrelationAnalyzer instance."""
        return EnergyCorrelationAnalyzer(n_sensors=4)

    def test_initialization(self, analyzer):
        """Test analyzer initializes correctly."""
        assert analyzer.n_sensors == 4

    def test_update(self, analyzer):
        """Test update with energy observations."""
        # update() takes sensor_id and energy individually
        analyzer.update(0, 1e-7)
        analyzer.update(1, 2e-7)
        analyzer.update(2, 1.5e-7)
        analyzer.update(3, 1e-7)

        assert len(analyzer.energy_history) > 0


class TestNashEquilibriumScheduler:
    """Tests for NashEquilibriumScheduler class."""

    @pytest.fixture
    def scheduler(self):
        """Create NashEquilibriumScheduler instance."""
        return NashEquilibriumScheduler(n_nodes=10)

    def test_initialization(self, scheduler):
        """Test scheduler initializes correctly."""
        assert scheduler.n_nodes == 10

    def test_compute_nash_probability(self, scheduler):
        """Test Nash probability computation."""
        # First update the energy estimate
        scheduler.update_energy_estimate(1e-6)

        # compute_nash_probability takes correlation_level, not expected_energy
        p = scheduler.compute_nash_probability(correlation_level=0.5)

        assert 0 <= p <= 1

    def test_nash_probability_capped_at_one(self, scheduler):
        """Test probability is capped at 1."""
        # Update with very high energy so probability reaches 1
        for _ in range(5):
            scheduler.update_energy_estimate(1.0)

        p = scheduler.compute_nash_probability(correlation_level=0.5)

        assert p <= 1.0

    def test_get_throughput_factor(self, scheduler):
        """Test throughput factor computation."""
        factor = scheduler.get_throughput_factor()

        expected = 1 - 1 / np.e
        assert_allclose(factor, expected, rtol=0.01)


class TestPredictiveMDPScheduler:
    """Tests for PredictiveMDPScheduler class."""

    @pytest.fixture
    def mdp(self):
        """Create PredictiveMDPScheduler instance."""
        return PredictiveMDPScheduler()

    def test_initialization(self, mdp):
        """Test MDP initializes correctly."""
        assert mdp is not None

    def test_predict_next_event(self, mdp):
        """Test event prediction."""
        # First train with some events
        for i, interval in enumerate([0.1, 0.12, 0.09, 0.11, 0.1]):
            mdp.update(event_time=float(i) * 0.1, amplitude=1e-6)

        # predict_next_event takes no arguments
        predicted_interval, predicted_amplitude = mdp.predict_next_event()

        assert predicted_interval > 0

    def test_compute_optimal_action(self, mdp):
        """Test optimal action computation."""
        # compute_optimal_action takes current_energy, time_since_last, and optionally transmission_energy
        action, value = mdp.compute_optimal_action(
            current_energy=1e-6, time_since_last=0.1, transmission_energy=1e-7
        )

        # Actual return values from implementation
        assert action in ["transmit_now", "wait", "sleep"]


class TestECCP:
    """Tests for ECCP class (Algorithm 3)."""

    @pytest.fixture
    def eccp(self):
        """Create ECCP instance."""
        return ECCP(n_sensors=10, transmission_energy=1e-7, sense_energy=1e-8)

    def test_initialization(self, eccp):
        """Test ECCP initializes correctly."""
        assert eccp.n_sensors == 10

    def test_on_wave_event(self, eccp):
        """Test wave event handling."""
        signal = np.sin(np.linspace(0, 10 * np.pi, 1000)) * 1e-6

        result = eccp.on_wave_event(node_id=0, timestamp=0.0, amplitude=1e-6, signal=signal)

        assert hasattr(result, "energy_harvested")
        assert result.energy_harvested >= 0

    def test_get_throughput_optimality(self, eccp):
        """Test throughput optimality computation."""
        optimality = eccp.get_throughput_optimality()

        assert 0 <= optimality <= 1


class TestECCPState:
    """Tests for ECCPState enum."""

    def test_state_values(self):
        """Test state enum values exist."""
        assert ECCPState.DEEP_SLEEP is not None
        assert ECCPState.SENSE is not None  # Not SENSING
        assert ECCPState.HARVEST is not None
        assert ECCPState.SYNC is not None


class TestECCPIntegration:
    """Integration tests for ECCP module."""

    def test_energy_harvesting_realistic(self):
        """Test energy harvesting with realistic parameters."""
        harvester = EnergyHarvester()

        # Typical steel structure vibration
        amplitude = 1e-6  # 1 micron
        frequency = 100  # 100 Hz

        power = harvester.compute_power(amplitude, frequency)

        # Should be positive and finite
        assert power >= 0
        assert np.isfinite(power)
