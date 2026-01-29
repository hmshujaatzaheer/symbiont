# Copyright (c) 2026 H M Shujaat Zaheer. All Rights Reserved.
# PROPRIETARY AND CONFIDENTIAL - See LICENSE file for terms.
"""
Core constants and physical parameters for SYMBIONT framework.

Based on the theoretical foundations from the proposal:
- Rose (2014): Ultrasonic Guided Waves in Solid Media
- Kay (1993): Fundamentals of Statistical Signal Processing
- Cover & Thomas (2005): Elements of Information Theory
"""

import numpy as np

# ==============================================================================
# Wave Propagation Constants (Definition 2: Structure as Synchronization Medium)
# ==============================================================================

# Steel wave velocities (Rose, 2014)
STEEL_LONGITUDINAL_VELOCITY = 5000.0  # m/s (compression waves)
STEEL_TRANSVERSE_VELOCITY = 3000.0  # m/s (shear waves)

# Concrete wave velocities
CONCRETE_LONGITUDINAL_VELOCITY = 4000.0  # m/s
CONCRETE_TRANSVERSE_VELOCITY = 2400.0  # m/s


# Effective velocity for multi-modal fusion (Theorem 3)
def compute_effective_velocity(c_L: float, c_T: float) -> float:
    """
    Compute effective velocity for multi-modal synchronization.

    From Theorem 3 (Multi-Modal Synchronization):
    c_eff = sqrt(c_L^2 + c_T^2)

    This provides sqrt(2) improvement in standard error.
    """
    return np.sqrt(c_L**2 + c_T**2)


STEEL_EFFECTIVE_VELOCITY = compute_effective_velocity(
    STEEL_LONGITUDINAL_VELOCITY, STEEL_TRANSVERSE_VELOCITY
)

# ==============================================================================
# SMIS Parameters (Section 3.2: Structure-Mediated Implicit Synchronization)
# ==============================================================================

DEFAULT_NOISE_STD = 0.0001  # 0.1ms measurement noise standard deviation
DEFAULT_SAMPLE_RATE = 10000  # Hz (10 kHz sampling)
DEFAULT_FFT_SIZE = 1024
DEFAULT_EVENT_THRESHOLD = 0.5  # Acceleration threshold for event detection (g)

# Gaussian Process hyperparameters for velocity field estimation
GP_LENGTH_SCALE = 10.0  # meters
GP_VARIANCE = 100.0  # (m/s)^2

# Temperature coefficient for velocity (approximately 0.1%/°C)
VELOCITY_TEMP_COEFFICIENT = 0.001  # per °C

# Velocity anomaly threshold for early warning (Proposition 1)
# Detectable at Δk/k ≈ 5% given 2-3% velocity resolution
VELOCITY_ANOMALY_THRESHOLD = 0.025  # 2.5% change

# ==============================================================================
# ECTI Parameters (Section 3.3: Edge-Computed Transmissibility Intelligence)
# ==============================================================================

# PINNA architecture parameters (Figure 5)
PINNA_FREQUENCY_BINS = 64  # K frequency bins
PINNA_CONV_FILTERS = 8  # Number of convolution filters
PINNA_DENSE_UNITS = 32  # Dense layer units
PINNA_DROPOUT_RATE = 0.2  # Dropout rate for MC Dropout
PINNA_TOTAL_PARAMS = 4127  # Total parameters
PINNA_MODEL_SIZE_BYTES = 4100  # INT8 quantized size

# Monte Carlo Dropout parameters (Gal & Ghahramani, 2016)
MC_DROPOUT_SAMPLES = 5  # M samples for uncertainty estimation

# Detection thresholds (Theorem 6: False Alarm Bound)
ANOMALY_SCORE_THRESHOLD = 0.5  # τ: anomaly score threshold
UNCERTAINTY_THRESHOLD = 0.2  # γ: uncertainty threshold

# Cross-pair consistency threshold
CONSISTENCY_THRESHOLD = 0.15  # τ_C

# Continual learning parameters
EMA_DECAY = 0.99  # β: exponential moving average decay
DRIFT_THRESHOLD = 0.1  # τ_drift: drift detection threshold
MEMORY_BUDGET_SAMPLES = 10  # Number of representative samples

# ==============================================================================
# ECCP Parameters (Section 3.4: Energy-Correlated Collaborative Protocol)
# ==============================================================================

# Piezoelectric harvester parameters (Proposition 2)
# MIDE V21BL piezoelectric coefficient
PIEZO_D31 = 190e-12  # m/V (piezoelectric strain constant)
PIEZO_VOLUME = 1e-6  # m^3 (approximate active volume)
PIEZO_INTERNAL_RESISTANCE = 100  # Ω

# Energy budget (Section 4.1.3)
ENERGY_SLEEP_POWER = 1e-6  # W (1 μW sleep)
ENERGY_SENSE_ECTI = 0.00025  # J (0.25 mJ for sensing + ECTI)
ENERGY_TRANSMIT = 0.0002  # J (0.2 mJ for transmission)
ENERGY_MIN_OPERATION = 0.001  # J (1 mJ minimum for one cycle)

# Game theory parameters (Theorem 8: Nash Equilibrium)
COLLISION_PENALTY = 1.0  # L: collision penalty in utility function

# Coverage parameters (Theorem 9)
MIN_NODE_DENSITY = 0.1  # nodes per meter

# ==============================================================================
# TEFL Parameters (Section 3.5: Transmissibility-Embedding Federated Learning)
# ==============================================================================

# Embedding dimension (d < |W| for communication efficiency)
EMBEDDING_DIMENSION = 16  # d: embedding dimension

# Differential privacy parameters (Theorem 12)
DEFAULT_EPSILON = 1.0  # ε: privacy parameter
DEFAULT_DELTA = 1e-5  # δ: privacy failure probability
EMBEDDING_SENSITIVITY = 10.0  # Δe: embedding sensitivity (Lipschitz bound)

# Convergence parameters (Theorem 11)
DEFAULT_LEARNING_RATE = 0.01  # η: learning rate for DT update
MAX_STALENESS = 10  # τ_max: maximum staleness for async FL

# Hierarchical FL parameters (Theorem 15)
HIERARCHY_LEVELS = 3  # Sensor → Gateway → Regional → Global

# ==============================================================================
# System Integration Parameters (Section 4.1.5)
# ==============================================================================

# Timing budget
SMIS_LATENCY_MS = 1  # SMIS processing time
ECTI_LATENCY_MS = 15  # ECTI inference time
NASH_LATENCY_MS = 0.1  # Nash probability computation
TOTAL_LATENCY_MS = 20  # Total pipeline latency

# Memory budget (within 32KB SRAM)
SMIS_MEMORY_KB = 1.0
ECTI_MEMORY_KB = 4.1
TEFL_MEMORY_KB = 1.0
BUFFER_MEMORY_KB = 2.0
TOTAL_MEMORY_KB = 8.1
TARGET_MCU_SRAM_KB = 32.0  # STM32L4+ SRAM

# ==============================================================================
# Performance Targets (Table 4)
# ==============================================================================

TARGET_SYNC_ACCURACY_MS = 0.1  # < 0.1ms synchronization
TARGET_SYNC_POWER_MW = 0.5  # < 0.5mW power consumption
TARGET_DETECTION_F1 = 0.96  # > 0.96 F1 score
TARGET_FALSE_ALARM_RATE = 0.01  # < 1% false alarm rate
TARGET_FL_COMM_BYTES = 500  # < 0.5KB per round
