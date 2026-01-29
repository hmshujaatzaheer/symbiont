# SYMBIONT

[![CI](https://github.com/hmshujaatzaheer/symbiont/actions/workflows/ci.yml/badge.svg)](https://github.com/hmshujaatzaheer/symbiont/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/hmshujaatzaheer/symbiont/branch/main/graph/badge.svg)](https://codecov.io/gh/hmshujaatzaheer/symbiont)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**SYMBiotic Infrastructure mONitoring with sTructure-mediated Intelligence**

A novel framework for structural health monitoring (SHM) that transforms civil infrastructure into self-aware entities through edge-computing, federated learning, and energy-aware protocols.

## 🎯 Overview

SYMBIONT introduces a paradigm shift in structural health monitoring by treating the physical structure itself as the primary medium for sensor coordination, eliminating the need for conventional RF-based clock synchronization.

### Key Innovations

| Component | Innovation | Performance Target |
|-----------|------------|-------------------|
| **SMIS** | Structure-mediated synchronization using wave propagation | <0.1ms sync error |
| **ECTI** | Physics-informed neural architecture (PINNA) for edge inference | >0.96 F1-score, 4.1KB model |
| **ECCP** | Game-theoretic energy-aware MAC protocol | <1% false alarms |
| **TEFL** | Privacy-preserving federated learning with digital twins | <0.5KB communication |

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Components](#components)
- [API Reference](#api-reference)
- [Theoretical Foundations](#theoretical-foundations)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## 🚀 Installation

### From PyPI (recommended)

```bash
pip install symbiont
```

### From Source

```bash
git clone https://github.com/hmshujaatzaheer/symbiont.git
cd symbiont
pip install -e .
```

### Development Installation

```bash
pip install -e ".[dev]"
```

## ⚡ Quick Start

```python
from symbiont import SMIS, ECTI, ECCP, TEFL
from symbiont.core.constants import STEEL_LONGITUDINAL_VELOCITY
import numpy as np

# Initialize components
smis = SMIS(num_sensors=4, sample_rate=100000)
ecti = ECTI(sample_rate=10000, num_frequency_bins=64)
eccp = ECCP(num_nodes=4, sample_rate=10000)
tefl = TEFL(num_nodes=4, num_frequency_bins=64)

# Process wave event
signals = {i: np.random.randn(2048) for i in range(4)}
sensor_positions = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])

# SMIS: Synchronize sensors
sync_result = smis.process_event(signals, sensor_positions, event_time=0.0)
print(f"Sync error: {sync_result['sync_errors']} seconds")

# ECTI: Detect damage
damage_result = ecti.infer(signals)
print(f"Damage probability: {damage_result['damage_probability']:.2%}")

# ECCP: Energy-aware protocol
protocol_result = eccp.on_wave_event(signals, timestamp=0.0)
print(f"Energy harvested: {protocol_result['energy_harvested']}")

# TEFL: Federated learning update
for node_id in range(4):
    T_mag, T_phase = np.random.randn(64), np.random.randn(64)
    fl_result = tefl.local_update(node_id, T_mag, T_phase)
    print(f"Node {node_id} embedding shape: {fl_result['embedding'].shape}")
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SYMBIONT Framework                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────┐│
│  │    SMIS     │   │    ECTI     │   │    ECCP     │   │  TEFL   ││
│  │  Structure- │   │   Edge-     │   │   Energy-   │   │  Trans- ││
│  │  Mediated   │──▶│  Computed   │──▶│  Correlated │──▶│  missi- ││
│  │  Implicit   │   │  Transmis-  │   │  Collabo-   │   │  bility ││
│  │  Sync       │   │  sibility   │   │  rative     │   │  Embed- ││
│  │             │   │  Intel.     │   │  Protocol   │   │  ding FL││
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────┘│
│        │                 │                 │                │      │
│        ▼                 ▼                 ▼                ▼      │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │                    Physical Structure                        │  │
│  │    (Wave Propagation, Energy Harvesting, Transmissibility)   │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 📦 Components

### SMIS: Structure-Mediated Implicit Synchronization

Exploits wave propagation through the structure for clock synchronization without RF.

```python
from symbiont.smis import SMIS, MultiModalWaveDecomposer, VelocityFieldEstimator

# Multimodal wave decomposition
decomposer = MultiModalWaveDecomposer(
    c_L=5000,  # Longitudinal velocity (m/s)
    c_T=3000,  # Transverse velocity (m/s)
    sample_rate=100000
)

# Extract wave arrivals
t_L, t_T, snr_L, snr_T = decomposer.extract_modal_arrivals(signal)

# Fuse for √2 improvement
t_fused, improvement = decomposer.fuse_modal_arrivals(t_L, t_T, snr_L, snr_T)
```

**Key Theorems:**
- **Theorem 2**: SE(δ̂) = √2 σ / (c √N)
- **Theorem 3**: Multimodal fusion achieves √2 improvement

### ECTI: Edge-Computed Transmissibility Intelligence

Physics-informed neural network for damage detection running entirely on edge devices.

```python
from symbiont.ecti import ECTI, PINNA

# Create PINNA model (4,127 parameters, ~4.1KB INT8)
pinna = PINNA(num_frequency_bins=64, num_filters=8)

# Monte Carlo dropout inference for uncertainty
mean_prob, uncertainty = pinna.mc_inference(T_ij, T_ji, num_samples=5)
```

**Features:**
- Complex-valued convolutions preserving phase
- Reciprocity enforcement layer
- Resonance-aware attention
- MC Dropout for uncertainty quantification

### ECCP: Energy-Correlated Collaborative Protocol

Game-theoretic MAC protocol for energy-harvesting sensors.

```python
from symbiont.eccp import ECCP, NashEquilibriumScheduler, EnergyHarvester

# Energy harvesting from structural vibrations
harvester = EnergyHarvester(d31=190e-12, volume=1e-6)
power = harvester.compute_power(amplitude=0.001, frequency=100)

# Nash equilibrium transmission probability
scheduler = NashEquilibriumScheduler(num_nodes=10, energy_per_transmission=0.0002)
p_star = scheduler.compute_nash_probability(mean_energy=0.001)
```

**Key Theorems:**
- **Theorem 8**: p* = min(1, E[E_h] / (N × E_tx))
- **Theorem 9**: P(coverage) ≥ 1 - exp(-ρ × E[E_h] / E_min)

### TEFL: Transmissibility-Embedding Federated Learning

Privacy-preserving federated learning for global structural knowledge.

```python
from symbiont.tefl import TEFL, DifferentialPrivacy

# Initialize with (ε, δ)-differential privacy
tefl = TEFL(num_nodes=100, epsilon=1.0, delta=1e-5)

# Local update with privacy
result = tefl.local_update(node_id=0, T_magnitude=T_mag, T_phase=T_phase)
privatized_embedding = result["privatized"]  # DP-protected

# Gateway aggregation
global_model = tefl.gateway_aggregate(all_updates)
```

**Privacy Guarantees:**
- **Theorem 12**: σ_DP = √(2 ln(1.25/δ)) × Δe / ε
- Advanced composition for multi-round privacy

## 📚 API Reference

### Core Constants

```python
from symbiont.core.constants import (
    STEEL_LONGITUDINAL_VELOCITY,  # 5000 m/s
    STEEL_TRANSVERSE_VELOCITY,    # 3000 m/s
    PINNA_TOTAL_PARAMS,           # 4127
    EMBEDDING_DIMENSION,          # 16
    DEFAULT_EPSILON,              # 1.0
    DEFAULT_DELTA,                # 1e-5
)
```

### Theorems

All 17 theorems from the research proposal are implemented:

```python
from symbiont.core.theorems import (
    mutual_information_through_structure,    # Theorem 1
    smis_synchronization_bound,              # Theorem 2
    multimodal_synchronization_error,        # Theorem 3
    cramer_rao_fisher_information,           # Theorem 4
    optimal_dispersion_fusion_variance,      # Theorem 5
    velocity_damage_relationship,            # Proposition 1
    false_alarm_bound,                       # Theorem 6
    damage_localization_check,               # Theorem 7
    harvested_power,                         # Proposition 2
    nash_equilibrium_probability,            # Theorem 8
    eccp_coverage_probability,               # Theorem 9
    eccp_throughput_factor,                  # Theorem 10
    tefl_convergence_bound,                  # Theorem 11
    differential_privacy_noise_scale,        # Theorem 12
    personalized_convergence_guaranteed,     # Theorem 13
    digital_twin_convergence_bound,          # Theorem 14
    hierarchical_communication_cost,         # Theorem 15
    async_convergence_bound,                 # Theorem 16
    tefl_optimal_time,                       # Theorem 17
)
```

## 🔬 Theoretical Foundations

### Key References

1. **Rose, J. L. (2014).** Ultrasonic Guided Waves in Solid Media. Cambridge University Press.
2. **Kay, S. M. (1993).** Fundamentals of Statistical Signal Processing: Estimation Theory.
3. **Dwork, C. (2014).** The Algorithmic Foundations of Differential Privacy.
4. **Han, S., et al. (2011).** Optimal energy allocation for wireless communications with energy harvesting constraints.
5. **McMahan, H. B., et al. (2017).** Communication-Efficient Learning of Deep Networks from Decentralized Data.
6. **Gal, Y. & Ghahramani, Z. (2016).** Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.
7. **Särkkä, S. (2013).** Bayesian Filtering and Smoothing. Cambridge University Press.

### Performance Targets

| Metric | Target | Component |
|--------|--------|-----------|
| Synchronization Error | <0.1 ms | SMIS |
| Detection F1-Score | >0.96 | ECTI |
| Model Size (INT8) | ~4.1 KB | ECTI/PINNA |
| False Alarm Rate | <1% | ECCP |
| Communication Budget | <0.5 KB | TEFL |

## 🔧 TinyML Deployment

This implementation provides the **algorithmic foundation** for SYMBIONT. For actual MCU deployment:

### Target Platform
- **MCU**: STM32L4+ (Cortex-M4F @ 48MHz)
- **Memory**: 32KB SRAM, 256KB Flash
- **Inference**: <15ms per event

### Deployment Workflow

```python
from symbiont.ecti import PINNA
from symbiont.tinyml import TinyMLExporter, MemoryAnalyzer

# 1. Train/validate PINNA model
pinna = PINNA(num_frequency_bins=64)

# 2. Export for TFLite conversion
exporter = TinyMLExporter()
model_dict = exporter.export_pinna(pinna)
exporter.save_for_tflite(model_dict, "pinna_model.json")

# 3. Analyze memory requirements
analyzer = MemoryAnalyzer()
analyzer.print_report()

# 4. For INT8 quantization, use TFLite converter externally:
#    tflite_convert --graph_def_file=pinna_model.json --output_file=pinna.tflite
#    
# 5. Or compile with STM32Cube.AI for STM32 deployment
```

### Memory Budget (from proposal)

| Component | Size |
|-----------|------|
| SMIS | 1.0 KB |
| ECTI/PINNA | 4.1 KB |
| TEFL encoder | 1.0 KB |
| Buffers | 2.0 KB |
| **Total** | **8.1 KB** |

**Note**: Actual TinyML deployment requires external toolchains (TensorFlow Lite Micro or STM32Cube.AI).

## 🧪 Testing

Run the complete test suite:

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=symbiont --cov-report=html

# Specific module
pytest tests/test_smis.py -v
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📖 Citation

If you use SYMBIONT in your research, please cite:

```bibtex
@phdthesis{zaheer2026symbiont,
  title={SYMBIONT: SYMBiotic Infrastructure mONitoring with sTructure-mediated Intelligence},
  author={H M Shujaat Zaheer},
  year={2026},
  school={[University Name]},
  note={PhD Proposal}
}
```

## 📄 License

Copyright © 2026 H M Shujaat Zaheer. All Rights Reserved.

This software is proprietary and confidential. See the [LICENSE](LICENSE) file for terms. Unauthorized copying, distribution, or use is strictly prohibited.

---

**SYMBIONT** - Transforming infrastructure into self-aware, collaborative entities.

*Copyright © 2026 H M Shujaat Zaheer. All Rights Reserved.*
