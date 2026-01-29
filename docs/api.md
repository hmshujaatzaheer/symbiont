# API Reference

## SMIS Module

### `symbiont.smis.SMIS`

Structure-Mediated Internal Synchronization (Algorithm 1).

```python
class SMIS(
    sensor_positions: np.ndarray,
    c_L: float = 5960.0,      # Longitudinal velocity (m/s)
    c_T: float = 3240.0,      # Transverse velocity (m/s)
    sample_rate: float = 100000.0,
    noise_std: float = 1e-6
)
```

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `synchronize_timestamps(raw_timestamps, reference_sensor)` | Synchronize sensor clocks | `SMISSyncResult` |
| `process_event(signals, event_time)` | Process wave event | `SMISEventResult` |
| `get_synchronization_bound(n_samples)` | Theoretical sync bound | `float` |

**Example:**
```python
from symbiont import SMIS
import numpy as np

positions = np.array([[0, 0], [5, 0], [10, 0]])
smis = SMIS(sensor_positions=positions)

result = smis.synchronize_timestamps(
    raw_timestamps={0: 0.0, 1: 0.00085, 2: 0.0017},
    reference_sensor=0
)
print(f"Error bound: {result.sync_error_bound}")
```

### `symbiont.smis.MultiModalWaveDecomposer`

Decomposes signals into longitudinal and transverse wave components.

```python
class MultiModalWaveDecomposer(
    sample_rate: float = 100000.0,
    c_L: float = 5960.0,
    c_T: float = 3240.0
)
```

### `symbiont.smis.VelocityFieldEstimator`

Bayesian velocity field estimation using Gaussian Processes.

```python
class VelocityFieldEstimator(
    sensor_positions: np.ndarray,
    initial_velocity: float = 5960.0,
    length_scale: float = 1.0,
    variance: float = 100.0
)
```

---

## ECTI Module

### `symbiont.ecti.ECTI`

Energy-aware Continual Transmissibility Inference (Algorithm 2).

```python
class ECTI(
    n_sensors: int,
    frequency_bins: int = 64,
    embedding_dim: int = 16
)
```

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `infer(signals, sensor_i, sensor_j)` | Run inference | `ECTIResult` |
| `detect_drift()` | Detect distribution drift | `bool` |
| `get_false_alarm_bound(p_score, p_uncertainty)` | Theorem 6 bound | `float` |

**Result Classes:**

```python
@dataclass
class ECTIResult:
    anomaly_score: float      # [0, 1] damage probability
    uncertainty: float        # Epistemic uncertainty
    embedding: np.ndarray     # Learned representation
    flag: ECTIFlag           # Detection flag

class ECTIFlag(Enum):
    NORMAL = "normal"
    STRUCTURAL_ANOMALY = "structural_anomaly"
    SENSOR_FAULT = "sensor_fault"
    REQUEST_CONFIRMATION = "request_confirmation"
```

### `symbiont.ecti.PINNA`

Physics-Informed Neural Network Architecture.

```python
class PINNA(
    frequency_bins: int = 64,
    conv_filters: int = 8,
    dense_units: int = 16,
    dropout_rate: float = 0.1
)
```

### `symbiont.ecti.TransmissibilityComputer`

Compute transmissibility between sensor pairs.

```python
class TransmissibilityComputer(sample_rate: float = 100000.0)
```

---

## ECCP Module

### `symbiont.eccp.ECCP`

Energy-Correlated Collaborative Protocol (Algorithm 3).

```python
class ECCP(
    n_sensors: int,
    transmission_energy: float = 1e-6,
    sense_energy: float = 1e-7
)
```

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `on_wave_event(node_id, timestamp, amplitude, signal, callbacks)` | Handle event | `UnifiedEventResult` |
| `get_throughput_optimality()` | Theorem 10 factor | `float` |

**Result Classes:**

```python
@dataclass
class UnifiedEventResult:
    sync_correction: float
    anomaly_result: ECTIResult
    transmission_decision: TransmissionDecision
    energy_harvested: float

@dataclass  
class TransmissionDecision:
    should_transmit: bool
    probability: float
    scheduled_time: float
```

### `symbiont.eccp.EnergyHarvester`

Piezoelectric energy harvesting model.

```python
class EnergyHarvester(
    d31: float = 190e-12,     # Piezo coefficient
    volume: float = 1e-6,     # Active volume
    internal_resistance: float = 1e6
)
```

### `symbiont.eccp.NashEquilibriumScheduler`

Game-theoretic transmission scheduling.

```python
class NashEquilibriumScheduler(
    num_nodes: int,
    transmission_energy: float = 1e-6
)
```

---

## TEFL Module

### `symbiont.tefl.TEFL`

Transmissibility Embedding Federated Learning (Algorithm 4).

```python
class TEFL(
    n_sensors: int,
    adjacency: np.ndarray,    # Network topology
    embedding_dim: int = 16,
    epsilon: float = 1.0,     # Privacy budget
    delta: float = 1e-5
)
```

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `local_update(node_id, signals, labels)` | Train locally | `np.ndarray` |
| `aggregate(embeddings)` | Hierarchical aggregation | `np.ndarray` |
| `get_privacy_guarantee()` | (ε, δ)-DP guarantee | `Tuple[float, float]` |

### `symbiont.tefl.DifferentialPrivacy`

Gaussian mechanism for differential privacy.

```python
class DifferentialPrivacy(
    epsilon: float = 1.0,
    delta: float = 1e-5,
    sensitivity: float = 1.0
)
```

### `symbiont.tefl.HierarchicalAggregator`

Communication-efficient hierarchical aggregation.

```python
class HierarchicalAggregator(
    n_levels: int = 3,
    embedding_dim: int = 16
)
```

---

## Core Module

### `symbiont.core.theorems`

All 17 theorems and 2 propositions from the paper.

| Function | Description |
|----------|-------------|
| `mutual_information_through_structure()` | Theorem 1: Channel capacity |
| `smis_synchronization_bound()` | Theorem 2: √n sync scaling |
| `multimodal_synchronization_error()` | Theorem 3: Multi-modal improvement |
| `cramer_rao_fisher_information()` | Theorem 4: CRLB |
| `optimal_dispersion_fusion()` | Theorem 5: Optimal fusion |
| `velocity_damage_sensitivity()` | Proposition 1: Damage detection |
| `false_alarm_probability_bound()` | Theorem 6: FA bound |
| `damage_localization()` | Theorem 7: Localization |
| `harvested_power()` | Proposition 2: Energy model |
| `nash_equilibrium_probability()` | Theorem 8: Nash equilibrium |
| `eccp_coverage_probability()` | Theorem 9: Coverage |
| `eccp_throughput_factor()` | Theorem 10: Throughput |
| `tefl_convergence_bound()` | Theorem 11: FL convergence |
| `dp_noise_scale()` | Theorem 12: DP noise |
| `personalized_convergence()` | Theorem 13: Non-IID handling |
| `digital_twin_convergence()` | Theorem 14: Twin sync |
| `hierarchical_communication_cost()` | Theorem 15: Comm savings |
| `async_convergence_bound()` | Theorem 16: Async FL |
| `tefl_optimal_time()` | Theorem 17: Optimal rounds |

### `symbiont.core.constants`

Physical and system constants.

```python
# Wave propagation
STEEL_LONGITUDINAL_VELOCITY = 5960.0  # m/s
STEEL_TRANSVERSE_VELOCITY = 3240.0    # m/s

# SMIS parameters
DEFAULT_SAMPLE_RATE = 100000.0  # Hz
DEFAULT_NOISE_STD = 1e-6        # seconds

# ECTI parameters
PINNA_FREQUENCY_BINS = 64
PINNA_MODEL_SIZE_BYTES = 4096

# ECCP parameters
ENERGY_TRANSMIT = 1e-6    # Joules
ENERGY_SENSE_ECTI = 1e-7  # Joules

# TEFL parameters
EMBEDDING_DIMENSION = 16
DEFAULT_EPSILON = 1.0
DEFAULT_DELTA = 1e-5
```
