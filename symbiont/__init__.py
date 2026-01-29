# Copyright (c) 2026 H M Shujaat Zaheer. All Rights Reserved.
# PROPRIETARY AND CONFIDENTIAL - See LICENSE file for terms.
"""
SYMBIONT: SYMBiotic Infrastructure mONitoring with sTructure-mediated Intelligence

A framework that exploits the monitored structure as an integral component of the
distributed sensing, computing, and communication system for Structural Health Monitoring.

Components:
- SMIS: Structure-Mediated Implicit Synchronization
- ECTI: Edge-Computed Transmissibility Intelligence
- ECCP: Energy-Correlated Collaborative Protocol
- TEFL: Transmissibility-Embedding Federated Learning

Theoretical Foundation:
    17 Theorems + 2 Propositions rigorously proven in the PhD proposal

TinyML Deployment:
    Target: STM32L4+ (Cortex-M4F @ 48MHz, 32KB SRAM)
    Model size: 4.1KB INT8 quantized (PINNA)
    Inference: <15ms per event

    For actual MCU deployment, use symbiont.tinyml module to export
    models for TensorFlow Lite Micro or STM32Cube.AI toolchains.

Author: H M Shujaat Zaheer
Email: shujabis@gmail.com
"""

__version__ = "1.0.0"
__author__ = "H M Shujaat Zaheer"
__email__ = "shujabis@gmail.com"

from symbiont.core.constants import (
    DEFAULT_NOISE_STD,
    DEFAULT_SAMPLE_RATE,
    STEEL_LONGITUDINAL_VELOCITY,
    STEEL_TRANSVERSE_VELOCITY,
)
from symbiont.eccp import ECCP, EnergyCorrelationAnalyzer, NashEquilibriumScheduler
from symbiont.ecti import ECTI, PINNA, TransmissibilityComputer
from symbiont.smis import SMIS, MultiModalWaveDecomposer, VelocityFieldEstimator
from symbiont.tefl import TEFL, DifferentialPrivacy, DigitalTwinSynchronizer

# TinyML deployment utilities
from symbiont.tinyml import MemoryAnalyzer, TinyMLExporter, verify_deployment_feasibility

__all__ = [
    # Core constants
    "STEEL_LONGITUDINAL_VELOCITY",
    "STEEL_TRANSVERSE_VELOCITY",
    "DEFAULT_NOISE_STD",
    "DEFAULT_SAMPLE_RATE",
    # SMIS components
    "SMIS",
    "MultiModalWaveDecomposer",
    "VelocityFieldEstimator",
    # ECTI components
    "ECTI",
    "PINNA",
    "TransmissibilityComputer",
    # ECCP components
    "ECCP",
    "NashEquilibriumScheduler",
    "EnergyCorrelationAnalyzer",
    # TEFL components
    "TEFL",
    "DifferentialPrivacy",
    "DigitalTwinSynchronizer",
    # TinyML deployment
    "TinyMLExporter",
    "MemoryAnalyzer",
    "verify_deployment_feasibility",
]
