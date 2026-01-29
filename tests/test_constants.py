"""
Comprehensive tests for SYMBIONT core constants module.
Ensures 100% code coverage of all physical parameters and utility functions.
"""

import numpy as np

from symbiont.core import constants


class TestWavePropagationConstants:
    """Tests for wave propagation physical constants."""

    def test_steel_longitudinal_velocity(self):
        """Verify steel longitudinal wave velocity (Rose, 2014)."""
        assert constants.STEEL_LONGITUDINAL_VELOCITY == 5000.0

    def test_steel_transverse_velocity(self):
        """Verify steel transverse wave velocity (Rose, 2014)."""
        assert constants.STEEL_TRANSVERSE_VELOCITY == 3000.0

    def test_concrete_longitudinal_velocity(self):
        """Verify concrete longitudinal wave velocity."""
        assert constants.CONCRETE_LONGITUDINAL_VELOCITY == 4000.0

    def test_concrete_transverse_velocity(self):
        """Verify concrete transverse wave velocity."""
        assert constants.CONCRETE_TRANSVERSE_VELOCITY == 2400.0

    def test_compute_effective_velocity(self):
        """Test effective velocity computation (Theorem 3)."""
        c_eff = constants.compute_effective_velocity(5000.0, 3000.0)
        expected = np.sqrt(5000.0**2 + 3000.0**2)
        assert np.isclose(c_eff, expected)

    def test_compute_effective_velocity_zero_inputs(self):
        """Test effective velocity with zero inputs."""
        c_eff = constants.compute_effective_velocity(0.0, 0.0)
        assert c_eff == 0.0

    def test_steel_effective_velocity_precomputed(self):
        """Verify precomputed steel effective velocity."""
        expected = constants.compute_effective_velocity(
            constants.STEEL_LONGITUDINAL_VELOCITY, constants.STEEL_TRANSVERSE_VELOCITY
        )
        assert np.isclose(constants.STEEL_EFFECTIVE_VELOCITY, expected)


class TestSMISParameters:
    """Tests for SMIS module parameters."""

    def test_default_noise_std(self):
        """Verify default noise standard deviation (0.1ms)."""
        assert constants.DEFAULT_NOISE_STD == 0.0001

    def test_default_sample_rate(self):
        """Verify default sample rate (10 kHz)."""
        assert constants.DEFAULT_SAMPLE_RATE == 10000

    def test_default_fft_size(self):
        """Verify default FFT size."""
        assert constants.DEFAULT_FFT_SIZE == 1024

    def test_default_event_threshold(self):
        """Verify event detection threshold."""
        assert constants.DEFAULT_EVENT_THRESHOLD == 0.5

    def test_gp_length_scale(self):
        """Verify Gaussian Process length scale."""
        assert constants.GP_LENGTH_SCALE == 10.0

    def test_gp_variance(self):
        """Verify Gaussian Process variance."""
        assert constants.GP_VARIANCE == 100.0

    def test_velocity_temp_coefficient(self):
        """Verify velocity temperature coefficient (0.1%/°C)."""
        assert constants.VELOCITY_TEMP_COEFFICIENT == 0.001

    def test_velocity_anomaly_threshold(self):
        """Verify velocity anomaly threshold (Proposition 1)."""
        assert constants.VELOCITY_ANOMALY_THRESHOLD == 0.025


class TestECTIParameters:
    """Tests for ECTI module parameters."""

    def test_pinna_frequency_bins(self):
        """Verify PINNA frequency bins."""
        assert constants.PINNA_FREQUENCY_BINS == 64

    def test_pinna_conv_filters(self):
        """Verify PINNA convolution filters."""
        assert constants.PINNA_CONV_FILTERS == 8

    def test_pinna_dense_units(self):
        """Verify PINNA dense layer units."""
        assert constants.PINNA_DENSE_UNITS == 32

    def test_pinna_dropout_rate(self):
        """Verify PINNA dropout rate."""
        assert constants.PINNA_DROPOUT_RATE == 0.2

    def test_pinna_total_params(self):
        """Verify PINNA total parameters (4,127)."""
        assert constants.PINNA_TOTAL_PARAMS == 4127

    def test_pinna_model_size_bytes(self):
        """Verify PINNA model size (<5KB INT8)."""
        assert constants.PINNA_MODEL_SIZE_BYTES == 4100
        assert constants.PINNA_MODEL_SIZE_BYTES < 5000

    def test_mc_dropout_samples(self):
        """Verify MC Dropout samples (Gal & Ghahramani, 2016)."""
        assert constants.MC_DROPOUT_SAMPLES == 5

    def test_anomaly_score_threshold(self):
        """Verify anomaly score threshold (Theorem 6)."""
        assert constants.ANOMALY_SCORE_THRESHOLD == 0.5

    def test_uncertainty_threshold(self):
        """Verify uncertainty threshold."""
        assert constants.UNCERTAINTY_THRESHOLD == 0.2

    def test_consistency_threshold(self):
        """Verify cross-pair consistency threshold."""
        assert constants.CONSISTENCY_THRESHOLD == 0.15

    def test_ema_decay(self):
        """Verify EMA decay for continual learning."""
        assert constants.EMA_DECAY == 0.99

    def test_drift_threshold(self):
        """Verify drift detection threshold."""
        assert constants.DRIFT_THRESHOLD == 0.1

    def test_memory_budget_samples(self):
        """Verify memory budget samples."""
        assert constants.MEMORY_BUDGET_SAMPLES == 10


class TestECCPParameters:
    """Tests for ECCP module parameters."""

    def test_piezo_d31(self):
        """Verify piezoelectric strain constant."""
        assert constants.PIEZO_D31 == 190e-12

    def test_piezo_volume(self):
        """Verify piezoelectric volume."""
        assert constants.PIEZO_VOLUME == 1e-6

    def test_piezo_internal_resistance(self):
        """Verify piezoelectric internal resistance."""
        assert constants.PIEZO_INTERNAL_RESISTANCE == 100

    def test_energy_sleep_power(self):
        """Verify sleep power (1 μW)."""
        assert constants.ENERGY_SLEEP_POWER == 1e-6

    def test_energy_sense_ecti(self):
        """Verify sense + ECTI energy (0.25 mJ)."""
        assert constants.ENERGY_SENSE_ECTI == 0.00025

    def test_energy_transmit(self):
        """Verify transmission energy (0.2 mJ)."""
        assert constants.ENERGY_TRANSMIT == 0.0002

    def test_energy_min_operation(self):
        """Verify minimum operation energy (1 mJ)."""
        assert constants.ENERGY_MIN_OPERATION == 0.001

    def test_collision_penalty(self):
        """Verify collision penalty (Theorem 8)."""
        assert constants.COLLISION_PENALTY == 1.0

    def test_min_node_density(self):
        """Verify minimum node density (Theorem 9)."""
        assert constants.MIN_NODE_DENSITY == 0.1


class TestTEFLParameters:
    """Tests for TEFL module parameters."""

    def test_embedding_dimension(self):
        """Verify embedding dimension."""
        assert constants.EMBEDDING_DIMENSION == 16

    def test_default_epsilon(self):
        """Verify default epsilon (Theorem 12)."""
        assert constants.DEFAULT_EPSILON == 1.0

    def test_default_delta(self):
        """Verify default delta (Theorem 12)."""
        assert constants.DEFAULT_DELTA == 1e-5

    def test_embedding_sensitivity(self):
        """Verify embedding sensitivity."""
        assert constants.EMBEDDING_SENSITIVITY == 10.0

    def test_default_learning_rate(self):
        """Verify default learning rate."""
        assert constants.DEFAULT_LEARNING_RATE == 0.01

    def test_max_staleness(self):
        """Verify maximum staleness (Theorem 16)."""
        assert constants.MAX_STALENESS == 10

    def test_hierarchy_levels(self):
        """Verify hierarchy levels (Theorem 15)."""
        assert constants.HIERARCHY_LEVELS == 3


class TestIntegrationParameters:
    """Tests for system integration parameters."""

    def test_smis_latency(self):
        """Verify SMIS latency."""
        assert constants.SMIS_LATENCY_MS == 1

    def test_ecti_latency(self):
        """Verify ECTI latency."""
        assert constants.ECTI_LATENCY_MS == 15

    def test_nash_latency(self):
        """Verify Nash computation latency."""
        assert constants.NASH_LATENCY_MS == 0.1

    def test_total_latency(self):
        """Verify total latency is under 20ms."""
        assert constants.TOTAL_LATENCY_MS == 20

    def test_smis_memory(self):
        """Verify SMIS memory."""
        assert constants.SMIS_MEMORY_KB == 1.0

    def test_ecti_memory(self):
        """Verify ECTI memory."""
        assert constants.ECTI_MEMORY_KB == 4.1

    def test_tefl_memory(self):
        """Verify TEFL memory."""
        assert constants.TEFL_MEMORY_KB == 1.0

    def test_buffer_memory(self):
        """Verify buffer memory."""
        assert constants.BUFFER_MEMORY_KB == 2.0

    def test_total_memory(self):
        """Verify total memory is under 32KB SRAM."""
        assert constants.TOTAL_MEMORY_KB == 8.1
        assert constants.TOTAL_MEMORY_KB < 32.0


class TestPerformanceTargets:
    """Tests for performance targets (Table 4)."""

    def test_target_sync_accuracy(self):
        """Verify sync accuracy target (<0.1ms)."""
        assert constants.TARGET_SYNC_ACCURACY_MS == 0.1

    def test_target_sync_power(self):
        """Verify sync power target (<0.5mW)."""
        assert constants.TARGET_SYNC_POWER_MW == 0.5

    def test_target_detection_f1(self):
        """Verify detection F1 target (>0.96)."""
        assert constants.TARGET_DETECTION_F1 == 0.96

    def test_target_false_alarm_rate(self):
        """Verify false alarm rate target (<1%)."""
        assert constants.TARGET_FALSE_ALARM_RATE == 0.01

    def test_target_fl_comm_bytes(self):
        """Verify FL communication target (<0.5KB)."""
        assert constants.TARGET_FL_COMM_BYTES == 500
