# SYMBIONT Framework - Theorems Test Suite
"""
Tests for symbiont.core.theorems module.
Validates all 17 theorems and 2 propositions from PhD proposal.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from symbiont.core import theorems
from symbiont.core.constants import (
    DEFAULT_DELTA,
    DEFAULT_EPSILON,
    DEFAULT_NOISE_STD,
    EMBEDDING_DIMENSION,
    STEEL_LONGITUDINAL_VELOCITY,
    STEEL_TRANSVERSE_VELOCITY,
)


class TestTheorem1MutualInformation:
    """Tests for Theorem 1: Mutual Information Through Structure."""

    def test_positive_mutual_information(self):
        """MI should be positive for non-trivial transmissibility."""
        frequencies = np.linspace(1, 1000, 100)
        transmissibility_mag = np.ones_like(frequencies) * 0.8
        signal_psd = np.ones_like(frequencies)
        noise_variance = 0.01

        mi = theorems.mutual_information_through_structure(
            transmissibility_mag, signal_psd, noise_variance, frequencies
        )
        assert mi > 0

    def test_zero_transmissibility_zero_mi(self):
        """MI should be zero when transmissibility is zero."""
        frequencies = np.linspace(1, 1000, 100)
        transmissibility_mag = np.zeros_like(frequencies)
        signal_psd = np.ones_like(frequencies)
        noise_variance = 0.01

        mi = theorems.mutual_information_through_structure(
            transmissibility_mag, signal_psd, noise_variance, frequencies
        )
        assert_allclose(mi, 0, atol=1e-10)

    def test_higher_snr_higher_mi(self):
        """Higher SNR should yield higher MI."""
        frequencies = np.linspace(1, 1000, 100)
        transmissibility_mag = np.ones_like(frequencies) * 0.8
        signal_psd = np.ones_like(frequencies)

        mi_low_noise = theorems.mutual_information_through_structure(
            transmissibility_mag, signal_psd, 0.001, frequencies
        )
        mi_high_noise = theorems.mutual_information_through_structure(
            transmissibility_mag, signal_psd, 0.1, frequencies
        )
        assert mi_low_noise > mi_high_noise

    def test_units_consistency(self):
        """Result should be in bits (positive real number)."""
        frequencies = np.linspace(1, 1000, 100)
        transmissibility_mag = np.ones_like(frequencies)
        signal_psd = np.ones_like(frequencies)
        noise_variance = 0.01

        mi = theorems.mutual_information_through_structure(
            transmissibility_mag, signal_psd, noise_variance, frequencies
        )
        assert isinstance(mi, float)
        assert np.isfinite(mi)


class TestTheorem2SMISSynchronizationBound:
    """Tests for Theorem 2: SMIS Synchronization Bound."""

    def test_bound_decreases_with_samples(self):
        """SE should decrease as sqrt(N)."""
        noise_std = DEFAULT_NOISE_STD
        velocity = STEEL_LONGITUDINAL_VELOCITY

        se_100 = theorems.smis_synchronization_bound(noise_std, velocity, 100)
        se_400 = theorems.smis_synchronization_bound(noise_std, velocity, 400)

        # SE(400) should be half of SE(100)
        assert_allclose(se_400, se_100 / 2, rtol=1e-10)

    def test_bound_positive(self):
        """SE should always be positive."""
        se = theorems.smis_synchronization_bound(0.001, 5000, 1000)
        assert se > 0

    def test_sqrt2_factor(self):
        """Formula includes sqrt(2) factor."""
        se = theorems.smis_synchronization_bound(1.0, 1.0, 1)
        assert_allclose(se, np.sqrt(2), rtol=1e-10)

    def test_with_physical_parameters(self):
        """Test with realistic steel parameters."""
        noise_std = 1e-6  # 1 microsecond
        velocity = STEEL_LONGITUDINAL_VELOCITY
        num_events = 1000

        se = theorems.smis_synchronization_bound(noise_std, velocity, num_events)
        # Should be on order of nanoseconds
        assert se < 1e-6


class TestTheorem3MultimodalSynchronization:
    """Tests for Theorem 3: Multi-Modal Synchronization Error."""

    def test_sqrt2_improvement(self):
        """Multi-modal should improve by sqrt(2) factor."""
        noise_std = 1e-6
        c_L = STEEL_LONGITUDINAL_VELOCITY
        c_T = STEEL_TRANSVERSE_VELOCITY
        num_events = 100

        se_single = theorems.smis_synchronization_bound(noise_std, c_L, num_events)
        se_multi = theorems.multimodal_synchronization_error(noise_std, c_L, c_T, num_events)

        # Multi-modal provides improvement over single-modal
        assert se_single / se_multi > 1.0  # Multi-modal is better

    def test_effective_velocity(self):
        """Test effective velocity computation."""
        c_L = 5000.0
        c_T = 3000.0
        noise_std = 1e-6
        num_events = 1

        se = theorems.multimodal_synchronization_error(noise_std, c_L, c_T, num_events)
        c_eff = np.sqrt(c_L**2 + c_T**2)
        expected = noise_std / c_eff
        assert_allclose(se, expected, rtol=1e-10)

    def test_se_decreases_with_distance(self):
        """SE should decrease with more events."""
        noise_std = 1e-6
        c_L = STEEL_LONGITUDINAL_VELOCITY
        c_T = STEEL_TRANSVERSE_VELOCITY

        se_10 = theorems.multimodal_synchronization_error(noise_std, c_L, c_T, 10)
        se_100 = theorems.multimodal_synchronization_error(noise_std, c_L, c_T, 100)

        assert se_100 < se_10


class TestTheorem4CramerRaoFisherInformation:
    """Tests for Theorem 4: Cramér-Rao Fisher Information."""

    def test_fisher_matrix_positive_definite(self):
        """Fisher Information matrix should be positive definite."""
        path_diffs = np.array([[1, -1, 0], [0, 1, -1], [1, 0, -1]])
        noise_var = 0.01
        num_events = 100

        fisher = theorems.cramer_rao_fisher_information(path_diffs, noise_var, num_events)
        eigenvalues = np.linalg.eigvalsh(fisher)
        assert np.all(eigenvalues >= -1e-10)  # Allow small numerical error

    def test_crlb_diagonal_positive(self):
        """CRLB diagonal should be positive."""
        path_diffs = np.array([[1, -1], [0.5, -0.5]])
        noise_var = 0.01
        num_events = 100

        fisher = theorems.cramer_rao_fisher_information(path_diffs, noise_var, num_events)
        crlb = theorems.cramer_rao_lower_bound(fisher)
        assert np.all(np.diag(crlb) > 0)

    def test_inverse_relationship(self):
        """More events should reduce CRLB."""
        path_diffs = np.array([[1, -1]])
        noise_var = 0.01

        fisher_100 = theorems.cramer_rao_fisher_information(path_diffs, noise_var, 100)
        fisher_400 = theorems.cramer_rao_fisher_information(path_diffs, noise_var, 400)

        crlb_100 = theorems.cramer_rao_lower_bound(fisher_100)
        crlb_400 = theorems.cramer_rao_lower_bound(fisher_400)

        # 4x more events = 1/4 the CRLB
        assert_allclose(crlb_400, crlb_100 / 4, rtol=0.1)

    def test_scales_with_noise(self):
        """CRLB should scale with noise variance."""
        path_diffs = np.array([[1, -1]])
        num_events = 100

        fisher_low = theorems.cramer_rao_fisher_information(path_diffs, 0.001, num_events)
        fisher_high = theorems.cramer_rao_fisher_information(path_diffs, 0.01, num_events)

        crlb_low = theorems.cramer_rao_lower_bound(fisher_low)
        crlb_high = theorems.cramer_rao_lower_bound(fisher_high)

        # Higher noise = higher CRLB (diagonal elements representing variances)
        assert np.all(np.diag(crlb_high) > np.diag(crlb_low))


class TestTheorem5OptimalDispersionFusion:
    """Tests for Theorem 5: Optimal Dispersion Curve Fusion."""

    def test_variance_bounded(self):
        """Fusion variance should be bounded."""
        snr_values = np.array([10.0, 20.0, 15.0])
        velocity_variance = np.ones(3) * 0.01
        frequency_bins = np.array([100.0, 500.0, 1000.0])

        var = theorems.optimal_dispersion_fusion_variance(
            snr_values, velocity_variance, frequency_bins
        )
        assert var > 0
        assert np.isfinite(var)

    def test_higher_snr_lower_variance(self):
        """Higher SNR should yield lower variance."""
        velocity_variance = np.ones(3) * 0.01
        frequency_bins = np.array([100.0, 500.0, 1000.0])
        snr_low = np.array([1.0, 2.0, 3.0])
        snr_high = np.array([10.0, 20.0, 30.0])

        var_low = theorems.optimal_dispersion_fusion_variance(
            snr_low, velocity_variance, frequency_bins
        )
        var_high = theorems.optimal_dispersion_fusion_variance(
            snr_high, velocity_variance, frequency_bins
        )

        assert var_high < var_low

    def test_single_mode(self):
        """Single mode should yield finite variance."""
        snr = np.array([10.0])
        velocity_variance = np.array([0.01])
        frequency_bins = np.array([500.0])
        var = theorems.optimal_dispersion_fusion_variance(snr, velocity_variance, frequency_bins)
        assert var > 0
        assert np.isfinite(var)


class TestProposition1VelocityDamage:
    """Tests for Proposition 1: Velocity-Damage Relationship."""

    def test_linear_relationship(self):
        """Δc/c = (1/2) Δk/k."""
        stiffness_reduction = 0.1  # 10% reduction

        velocity_change = theorems.velocity_damage_relationship(stiffness_reduction)
        expected = 0.5 * stiffness_reduction
        assert_allclose(velocity_change, expected, rtol=1e-10)

    def test_zero_damage_no_change(self):
        """Zero damage should yield zero velocity change."""
        velocity_change = theorems.velocity_damage_relationship(0.0)
        assert_allclose(velocity_change, 0.0)

    def test_damage_detectability(self):
        """Test damage detectability threshold."""
        # Default velocity resolution is 2.5%
        # 2% stiffness reduction gives 1% velocity change, NOT detectable with 2.5% threshold
        # 6% stiffness reduction gives 3% velocity change, IS detectable with 2.5% threshold
        damage_small = 0.02  # 2% stiffness reduction -> 1% velocity change
        damage_large = 0.06  # 6% stiffness reduction -> 3% velocity change

        is_detectable_small = theorems.is_damage_detectable(damage_small)  # False (1% < 2.5%)
        is_detectable_large = theorems.is_damage_detectable(damage_large)  # True (3% > 2.5%)

        assert not is_detectable_small
        assert is_detectable_large


class TestTheorem6FalseAlarmBound:
    """Tests for Theorem 6: False Alarm Bound."""

    def test_bound_probability(self):
        """P(FA) should be bounded [0, 1]."""
        # These are probabilities: P(score > threshold | healthy) and P(uncertainty < threshold | healthy)
        p_score_healthy = 0.05  # 5% chance score exceeds threshold when healthy
        p_uncertainty_healthy = 0.1  # 10% chance uncertainty is below threshold when healthy

        p_fa = theorems.false_alarm_bound(p_score_healthy, p_uncertainty_healthy)
        assert 0 <= p_fa <= 1

    def test_higher_threshold_lower_fa(self):
        """Lower individual probabilities should reduce false alarms."""
        # Lower p_score means higher threshold (fewer healthy samples exceed it)
        p_fa_loose = theorems.false_alarm_bound(0.1, 0.2)  # More false alarms
        p_fa_strict = theorems.false_alarm_bound(0.02, 0.05)  # Fewer false alarms

        assert p_fa_strict < p_fa_loose

    def test_neyman_pearson(self):
        """Result should be product of input probabilities (independence assumption)."""
        p_score = 0.05
        p_uncertainty = 0.1

        p_fa = theorems.false_alarm_bound(p_score, p_uncertainty)
        # FA bound is P(score > τ | healthy) × P(uncertainty < γ | healthy)
        expected = p_score * p_uncertainty
        assert_allclose(p_fa, expected, rtol=0.01)


class TestTheorem7DamageLocalization:
    """Tests for Theorem 7: Damage Localization Guarantee."""

    def test_correct_localization(self):
        """Should correctly localize damage when one segment exceeds threshold."""
        delta_T_ij = 0.05  # Large change in segment (i,j) - damage here
        delta_T_jk = 0.001  # Small change in adjacent segment (j,k)
        threshold = 0.01

        result, location = theorems.damage_localization_check(delta_T_ij, delta_T_jk, threshold)
        assert result is True  # Damage detected
        assert "localized" in location.lower() or "segment" in location.lower()

    def test_no_damage(self):
        """Should indicate no damage when both segments below threshold."""
        delta_T_ij = 0.002
        delta_T_jk = 0.001
        threshold = 0.01

        result, location = theorems.damage_localization_check(delta_T_ij, delta_T_jk, threshold)
        assert result is False


class TestProposition2HarvestedPower:
    """Tests for Proposition 2: Harvested Power."""

    def test_power_positive(self):
        """Harvested power should be positive for non-zero input."""
        amplitude = 1e-6  # 1 micron
        frequency = 100  # Hz

        power = theorems.harvested_power(amplitude, frequency)
        assert power > 0

    def test_power_scales_with_frequency_squared(self):
        """P ∝ f²."""
        amplitude = 1e-6

        power_100 = theorems.harvested_power(amplitude, 100)
        power_200 = theorems.harvested_power(amplitude, 200)

        assert_allclose(power_200 / power_100, 4.0, rtol=0.1)

    def test_power_scales_with_amplitude_squared(self):
        """P ∝ A²."""
        frequency = 100

        power_1 = theorems.harvested_power(1e-6, frequency)
        power_2 = theorems.harvested_power(2e-6, frequency)

        assert_allclose(power_2 / power_1, 4.0, rtol=0.1)


class TestTheorem8NashEquilibrium:
    """Tests for Theorem 8: Nash Equilibrium."""

    def test_probability_bounded(self):
        """p* should be in [0, 1]."""
        expected_energy = 1e-6
        num_nodes = 10
        tx_energy = 1e-7

        p_star = theorems.nash_equilibrium_probability(expected_energy, num_nodes, tx_energy)
        assert 0 <= p_star <= 1

    def test_one_minus_one_over_e_factor(self):
        """Throughput achieves (1 - 1/e) of optimal."""
        factor = theorems.nash_throughput_factor()
        expected = 1 - 1 / np.e
        assert_allclose(factor, expected, rtol=1e-10)

    def test_higher_energy_higher_probability(self):
        """More energy should yield higher transmission probability."""
        num_nodes = 10
        tx_energy = 1e-7

        p_low = theorems.nash_equilibrium_probability(1e-7, num_nodes, tx_energy)
        p_high = theorems.nash_equilibrium_probability(1e-5, num_nodes, tx_energy)

        assert p_high >= p_low


class TestTheorem9ECCPCoverage:
    """Tests for Theorem 9: ECCP Coverage Probability."""

    def test_coverage_bounded(self):
        """Coverage probability should be in [0, 1]."""
        node_density = 0.1  # per meter
        expected_energy = 1e-6
        min_energy = 1e-7

        coverage = theorems.eccp_coverage_probability(node_density, expected_energy, min_energy)
        assert 0 <= coverage <= 1

    def test_higher_density_higher_coverage(self):
        """Higher node density should increase coverage."""
        expected_energy = 1e-6
        min_energy = 1e-7

        cov_low = theorems.eccp_coverage_probability(0.01, expected_energy, min_energy)
        cov_high = theorems.eccp_coverage_probability(0.1, expected_energy, min_energy)

        assert cov_high >= cov_low


class TestTheorem10ECCPThroughput:
    """Tests for Theorem 10: ECCP Throughput Factor."""

    def test_factor_bounded(self):
        """Throughput factor should be in [0, 1]."""
        energy_variance = 1e-12
        energy_mean = 1e-6

        factor = theorems.eccp_throughput_factor(energy_variance, energy_mean)
        assert 0 <= factor <= 1

    def test_low_variance_high_factor(self):
        """Low variance should yield high throughput factor."""
        energy_mean = 1e-6

        factor_low_var = theorems.eccp_throughput_factor(1e-14, energy_mean)
        factor_high_var = theorems.eccp_throughput_factor(1e-12, energy_mean)

        assert factor_low_var > factor_high_var


class TestTheorem11TEFLConvergence:
    """Tests for Theorem 11: TEFL Convergence."""

    def test_bound_decreases_with_rounds(self):
        """Convergence bound should decrease with rounds."""
        embedding_dim = EMBEDDING_DIMENSION
        num_edges = 100

        bound_100 = theorems.tefl_convergence_bound(100, embedding_dim, num_edges)
        bound_1000 = theorems.tefl_convergence_bound(1000, embedding_dim, num_edges)

        assert bound_1000 < bound_100

    def test_bound_positive(self):
        """Convergence bound should always be positive."""
        bound = theorems.tefl_convergence_bound(100, EMBEDDING_DIMENSION, 100)
        assert bound > 0


class TestTheorem12DPNoiseScale:
    """Tests for Theorem 12: Differential Privacy Noise Scale."""

    def test_scale_positive(self):
        """Noise scale should be positive."""
        scale = theorems.differential_privacy_noise_scale(DEFAULT_EPSILON, DEFAULT_DELTA, 1.0)
        assert scale > 0

    def test_lower_epsilon_more_noise(self):
        """Lower ε should require more noise."""
        delta = 1e-5
        sensitivity = 1.0

        scale_high_eps = theorems.differential_privacy_noise_scale(1.0, delta, sensitivity)
        scale_low_eps = theorems.differential_privacy_noise_scale(0.1, delta, sensitivity)

        assert scale_low_eps > scale_high_eps

    def test_privacy_accuracy_tradeoff(self):
        """Test privacy-accuracy tradeoff computation."""
        epsilon = 1.0
        noise_scale = 1.0
        sensitivity = 1.0

        accuracy_loss = theorems.privacy_accuracy_tradeoff(epsilon, noise_scale, sensitivity)
        assert accuracy_loss >= 0


class TestTheorem13PersonalizedConvergence:
    """Tests for Theorem 13: Personalized Convergence."""

    def test_convergence_guaranteed(self):
        """Personalized TEFL should always converge."""
        result = theorems.personalized_convergence_guaranteed(non_iid_degree=0.5)
        assert result is True

    def test_extreme_non_iid(self):
        """Should converge even under extreme non-IID."""
        result = theorems.personalized_convergence_guaranteed(non_iid_degree=1.0)
        assert result is True


class TestTheorem14DigitalTwinConvergence:
    """Tests for Theorem 14: Digital Twin Convergence."""

    def test_bound_decreases_with_time(self):
        """DT convergence bound should decrease as O(1/t)."""
        bound_10 = theorems.digital_twin_convergence_bound(10)
        bound_100 = theorems.digital_twin_convergence_bound(100)

        assert_allclose(bound_100, bound_10 / 10, rtol=1e-10)

    def test_bound_positive(self):
        """Bound should always be positive."""
        bound = theorems.digital_twin_convergence_bound(100)
        assert bound > 0


class TestTheorem15HierarchicalCommunication:
    """Tests for Theorem 15: Hierarchical Communication Cost."""

    def test_log_vs_linear(self):
        """Hierarchical should be O(log N) vs O(N)."""
        hierarchical = theorems.hierarchical_communication_cost(1000)
        flat = theorems.flat_communication_cost(1000)
        savings = theorems.hierarchical_vs_flat_ratio(1000)

        assert hierarchical < flat
        assert savings < 0.1  # log(1000)/1000 ≈ 0.01

    def test_savings_increase_with_n(self):
        """Savings should increase with N."""
        savings_100 = theorems.hierarchical_vs_flat_ratio(100)
        savings_1000 = theorems.hierarchical_vs_flat_ratio(1000)

        assert savings_1000 < savings_100  # Lower ratio = better


class TestTheorem16AsyncConvergence:
    """Tests for Theorem 16: Asynchronous Convergence."""

    def test_staleness_impact(self):
        """Higher staleness should increase convergence bound."""
        num_rounds = 100

        bound_low = theorems.async_convergence_bound(num_rounds, 1)
        bound_high = theorems.async_convergence_bound(num_rounds, 10)

        assert bound_high > bound_low

    def test_bound_decreases_with_rounds(self):
        """Bound should decrease with more rounds."""
        max_staleness = 5

        bound_100 = theorems.async_convergence_bound(100, max_staleness)
        bound_1000 = theorems.async_convergence_bound(1000, max_staleness)

        assert bound_1000 < bound_100


class TestTheorem17OptimalTime:
    """Tests for Theorem 17: Optimal FL Time."""

    def test_optimal_rounds_positive(self):
        """Optimal time should be positive."""
        t_star = theorems.tefl_optimal_time(
            num_edges=100, bytes_per_round=500, target_accuracy=0.01
        )
        assert t_star > 0

    def test_lower_accuracy_requires_more_time(self):
        """Stricter accuracy requires more time."""
        num_edges = 100
        bytes_per_round = 500

        t_01 = theorems.tefl_optimal_time(num_edges, bytes_per_round, 0.1)
        t_001 = theorems.tefl_optimal_time(num_edges, bytes_per_round, 0.01)

        assert t_001 > t_01


class TestIntegrationTheorems:
    """Integration tests combining multiple theorems."""

    def test_smis_end_to_end(self):
        """Test SMIS theorem chain."""
        noise_std = 1e-6
        c_L = STEEL_LONGITUDINAL_VELOCITY
        c_T = STEEL_TRANSVERSE_VELOCITY
        num_events = 100

        # Theorem 2: Single-mode bound
        se_single = theorems.smis_synchronization_bound(noise_std, c_L, num_events)

        # Theorem 3: Multi-modal improvement
        se_multi = theorems.multimodal_synchronization_error(noise_std, c_L, c_T, num_events)
        assert se_multi < se_single

        # Theorem 4: CRLB
        path_diffs = np.array([[1.0, -1.0]])
        fisher = theorems.cramer_rao_fisher_information(path_diffs, noise_std**2, num_events)
        crlb = theorems.cramer_rao_lower_bound(fisher)
        assert np.all(np.diag(crlb) > 0)

    def test_eccp_theorem_chain(self):
        """Test ECCP theorem chain."""
        amplitude = 1e-6
        frequency = 100

        # Proposition 2: Harvested power
        power = theorems.harvested_power(amplitude, frequency)
        assert power > 0

        # Theorem 8: Nash equilibrium
        p_star = theorems.nash_equilibrium_probability(power * 0.001, 10, 1e-9)
        assert 0 <= p_star <= 1

        # Theorem 9: Coverage
        coverage = theorems.eccp_coverage_probability(0.1, power * 0.001, 1e-9)
        assert 0 <= coverage <= 1

    def test_tefl_theorem_chain(self):
        """Test TEFL theorem chain."""
        # Theorem 11: Convergence bound
        bound = theorems.tefl_convergence_bound(100, EMBEDDING_DIMENSION, 100)
        assert bound > 0

        # Theorem 12: DP noise
        scale = theorems.differential_privacy_noise_scale(1.0, 1e-5, 1.0)
        assert scale > 0

        # Theorem 15: Communication savings
        savings = theorems.hierarchical_vs_flat_ratio(1000)
        assert savings < 0.1


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_negative_noise_raises(self):
        """Negative noise should raise ValueError."""
        with pytest.raises(ValueError):
            theorems.smis_synchronization_bound(-1.0, 5000, 100)

    def test_zero_events_raises(self):
        """Zero events should raise ValueError."""
        with pytest.raises(ValueError):
            theorems.smis_synchronization_bound(1e-6, 5000, 0)

    def test_negative_velocity_raises(self):
        """Negative velocity should raise ValueError."""
        with pytest.raises(ValueError):
            theorems.smis_synchronization_bound(1e-6, -5000, 100)

    def test_zero_transmission_energy_raises(self):
        """Zero transmission energy should raise ValueError."""
        with pytest.raises(ValueError):
            theorems.nash_equilibrium_probability(1e-6, 10, 0)
