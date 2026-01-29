# SYMBIONT Framework - Pytest Configuration
"""
Shared fixtures for SYMBIONT test suite.
"""

import numpy as np
import pytest

from symbiont.core.constants import (
    EMBEDDING_DIMENSION,
    PINNA_FREQUENCY_BINS,
    STEEL_LONGITUDINAL_VELOCITY,
)


@pytest.fixture
def random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    return 42


@pytest.fixture
def sensor_positions_2d():
    """Standard 2D sensor positions (2m x 2m grid)."""
    return np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ]
    )


@pytest.fixture
def sensor_positions_1d():
    """1D sensor positions along a beam."""
    return np.array([[i * 0.5] for i in range(5)])


@pytest.fixture
def sample_rate():
    """Standard sample rate for tests."""
    return 100000  # 100 kHz


@pytest.fixture
def synthetic_signals(sensor_positions_2d, sample_rate):
    """Generate synthetic wave signals."""
    num_sensors = len(sensor_positions_2d)
    num_samples = 2048
    event_position = np.array([1.0, 1.0])

    signals = {}
    for i in range(num_sensors):
        distance = np.linalg.norm(sensor_positions_2d[i] - event_position)
        arrival_time = distance / STEEL_LONGITUDINAL_VELOCITY
        arrival_sample = int(arrival_time * sample_rate)

        signal = np.zeros(num_samples)
        pulse_width = 50
        if arrival_sample < num_samples - pulse_width:
            pulse = np.exp(-0.5 * ((np.arange(pulse_width) - pulse_width // 2) / 10) ** 2)
            signal[arrival_sample : arrival_sample + pulse_width] = pulse

        signal += 0.01 * np.random.randn(num_samples)
        signals[i] = signal

    return signals


@pytest.fixture
def transmissibility_pair():
    """Generate transmissibility magnitude and phase."""
    T_magnitude = np.random.randn(PINNA_FREQUENCY_BINS)
    T_phase = np.random.randn(PINNA_FREQUENCY_BINS)
    return T_magnitude, T_phase


@pytest.fixture
def embedding():
    """Generate random embedding."""
    return np.random.randn(EMBEDDING_DIMENSION)


@pytest.fixture
def frequency_array():
    """Standard frequency array for transmissibility."""
    return np.linspace(10, 1000, PINNA_FREQUENCY_BINS)


# Markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks integration tests")
