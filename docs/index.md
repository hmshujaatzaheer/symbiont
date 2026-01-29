# SYMBIONT Documentation

**SYnchronized Multi-modal BIodegradable Oscillation-driven Networks for Telecommunications**

A Python framework for wireless structural health monitoring (SHM) sensor networks, implementing energy-aware protocols with theoretical guarantees.

## Overview

SYMBIONT addresses the fundamental challenge of deploying battery-free wireless sensor networks for infrastructure monitoring. The framework provides:

- **Sub-microsecond synchronization** without GPS using structural wave propagation
- **Uncertainty-aware damage detection** with bounded false alarm rates
- **Energy-harvesting aware MAC protocols** with Nash equilibrium guarantees
- **Privacy-preserving federated learning** for distributed anomaly detection

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      SYMBIONT Framework                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
│  │  SMIS   │  │  ECTI   │  │  ECCP   │  │  TEFL   │            │
│  │ Algo 1  │  │ Algo 2  │  │ Algo 3  │  │ Algo 4  │            │
│  │         │  │         │  │         │  │         │            │
│  │ Sync    │  │ Damage  │  │ Energy  │  │ Fed.    │            │
│  │ Protocol│  │ Detect  │  │ Protocol│  │ Learning│            │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘            │
│       │            │            │            │                  │
│       └────────────┴────────────┴────────────┘                  │
│                         │                                       │
│              ┌──────────┴──────────┐                           │
│              │   Core Theorems     │                           │
│              │   (17 + 2 Props)    │                           │
│              └─────────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

```python
import numpy as np
from symbiont import SMIS, ECTI, ECCP, TEFL

# Initialize sensor network
positions = np.array([[0, 0], [5, 0], [10, 0], [5, 5]])
smis = SMIS(sensor_positions=positions, sample_rate=100000)

# Synchronize clocks using wave propagation
sync_result = smis.synchronize_timestamps(
    raw_timestamps={0: 0.0, 1: 0.00085, 2: 0.0017, 3: 0.001},
    reference_sensor=0
)
print(f"Synchronization error: {sync_result.sync_error_bound*1e6:.2f} µs")
```

## Contents

- [Installation](installation.md)
- [API Reference](api.md)
- [Theoretical Foundations](theorems.md)
- [Examples](examples.md)
- [Contributing](contributing.md)

## Modules

| Module | Description | Key Features |
|--------|-------------|--------------|
| **SMIS** | Structure-Mediated Internal Synchronization | Multi-modal wave decomposition, Bayesian velocity estimation |
| **ECTI** | Energy-aware Continual Transmissibility Inference | PINNA neural network, MC-Dropout uncertainty, drift detection |
| **ECCP** | Energy-Correlated Collaborative Protocol | Piezoelectric harvesting, Nash equilibrium scheduling |
| **TEFL** | Transmissibility Embedding Federated Learning | Differential privacy, hierarchical aggregation, digital twins |

## Citation

```bibtex
@phdthesis{zaheer2026symbiont,
  title={SYMBIONT: SYMBiotic Infrastructure mONitoring with 
         sTructure-mediated Intelligence},
  author={H M Shujaat Zaheer},
  school={[University Name]},
  year={2026},
  note={PhD Proposal}
}
```

## License

Copyright © 2026 H M Shujaat Zaheer. All Rights Reserved. Proprietary License - see [LICENSE](../LICENSE) for details.
