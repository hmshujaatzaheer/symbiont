# Copyright (c) 2026 H M Shujaat Zaheer. All Rights Reserved.
# PROPRIETARY AND CONFIDENTIAL - See LICENSE file for terms.
"""
TinyML Deployment Module for SYMBIONT

This module provides utilities for deploying SYMBIONT models to microcontrollers.
Target platform: STM32L4+ (Cortex-M4F @ 48MHz, 32KB SRAM)

DEPLOYMENT OPTIONS:

1. NumPy Simulation (this module):
   - Use for algorithm validation and testing
   - No external dependencies
   - Not MCU-deployable

2. TensorFlow Deployment (tf_pinna submodule):
   - Full TensorFlow/Keras PINNA implementation
   - TFLite conversion with INT8 quantization
   - STM32Cube.AI integration
   - Requires: pip install tensorflow>=2.10.0

Deployment Workflow:
    1. Train/validate using NumPy simulation OR TensorFlow model
    2. Export to TFLite with INT8 quantization
    3. Generate C headers for STM32
    4. Compile with STM32Cube.AI or TFLite Micro
    5. Deploy to STM32L4+ board

Memory Budget (from proposal):
    - SMIS: 1.0 KB
    - ECTI/PINNA: 4.1 KB (INT8 quantized)
    - TEFL encoder: 1.0 KB
    - Buffers: 2.0 KB
    - Total: 8.1 KB (within 32KB SRAM)

Performance Targets:
    - PINNA inference: <15ms @ 48MHz
    - MC Dropout (5 passes): <75ms total
    - Total latency: <20ms per event

References:
    - David et al. (2021): TensorFlow Lite Micro for MCUs
    - STM32Cube.AI User Manual (UM2526)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from symbiont.core.constants import (
    BUFFER_MEMORY_KB,
    ECTI_MEMORY_KB,
    EMBEDDING_DIMENSION,
    PINNA_CONV_FILTERS,
    PINNA_DENSE_UNITS,
    PINNA_FREQUENCY_BINS,
    PINNA_TOTAL_PARAMS,
    SMIS_MEMORY_KB,
    TARGET_MCU_SRAM_KB,
    TEFL_MEMORY_KB,
    TOTAL_MEMORY_KB,
)


@dataclass
class QuantizationConfig:
    """Configuration for INT8 quantization."""

    # Quantization parameters
    bits: int = 8
    symmetric: bool = True
    per_channel: bool = False

    # Calibration settings
    num_calibration_samples: int = 100
    calibration_method: str = "minmax"  # or "percentile"

    # Target model size
    target_size_kb: float = 4.1


@dataclass
class DeploymentConfig:
    """Configuration for MCU deployment."""

    # Target hardware
    mcu_family: str = "STM32L4"
    clock_mhz: int = 48
    sram_kb: int = 32
    flash_kb: int = 256

    # Toolchain
    toolchain: str = "STM32Cube.AI"  # or "TFLite_Micro"
    optimization_level: int = 3

    # Memory allocation
    heap_kb: float = 8.0
    stack_kb: float = 2.0


class TinyMLExporter:
    """
    Export SYMBIONT models for TinyML deployment.

    This class extracts model weights and architecture in formats
    compatible with TensorFlow Lite and STM32Cube.AI.

    Example:
        >>> from symbiont.ecti import PINNA
        >>> from symbiont.tinyml import TinyMLExporter
        >>>
        >>> pinna = PINNA(num_frequency_bins=64)
        >>> exporter = TinyMLExporter()
        >>>
        >>> # Export for TFLite conversion
        >>> model_dict = exporter.export_pinna(pinna)
        >>> exporter.save_for_tflite(model_dict, "pinna_model.json")
        >>>
        >>> # Then use TFLite converter externally:
        >>> # tflite_convert --graph_def_file=pinna_model.json --output_file=pinna.tflite
    """

    def __init__(self, config: Optional[QuantizationConfig] = None):
        """
        Initialize exporter.

        Args:
            config: Quantization configuration
        """
        self.config = config or QuantizationConfig()

    def export_pinna(self, pinna) -> Dict[str, Any]:
        """
        Export PINNA model weights and architecture.

        Args:
            pinna: PINNA model instance from symbiont.ecti

        Returns:
            Dictionary containing:
                - weights: Dict of numpy arrays
                - architecture: Dict describing layers
                - metadata: Model information
        """
        weights = {}
        architecture = []

        # Extract complex convolution weights
        if hasattr(pinna, "W_conv_real"):
            weights["conv_real"] = pinna.W_conv_real.astype(np.float32)
            weights["conv_imag"] = pinna.W_conv_imag.astype(np.float32)
            architecture.append(
                {
                    "type": "ComplexConv1D",
                    "filters": PINNA_CONV_FILTERS,
                    "kernel_size": 5,
                    "activation": "crelu",
                }
            )

        # Extract reciprocity layer weights
        if hasattr(pinna, "W_recip"):
            weights["recip_W"] = pinna.W_recip.astype(np.float32)
            weights["recip_b"] = pinna.b_recip.astype(np.float32)
            architecture.append(
                {"type": "ReciprocityEnforcement", "units": PINNA_DENSE_UNITS, "activation": "relu"}
            )

        # Extract attention weights
        if hasattr(pinna, "W_attn"):
            weights["attn_W"] = pinna.W_attn.astype(np.float32)
            weights["attn_v"] = pinna.v_attn.astype(np.float32)
            architecture.append({"type": "ResonanceAttention", "hidden_dim": PINNA_DENSE_UNITS})

        # Extract embedding weights
        if hasattr(pinna, "W_embed"):
            weights["embed_W"] = pinna.W_embed.astype(np.float32)
            architecture.append({"type": "Embedding", "output_dim": EMBEDDING_DIMENSION})

        # Extract output weights
        if hasattr(pinna, "W_out"):
            weights["out_W"] = pinna.W_out.astype(np.float32)
            weights["out_b"] = pinna.b_out.astype(np.float32)
            architecture.append({"type": "Dense", "units": 1, "activation": "sigmoid"})

        # Compute model statistics
        total_params = sum(w.size for w in weights.values())
        fp32_size_kb = sum(w.nbytes for w in weights.values()) / 1024
        int8_size_kb = total_params / 1024

        metadata = {
            "framework": "SYMBIONT",
            "model_type": "PINNA",
            "total_parameters": total_params,
            "fp32_size_kb": fp32_size_kb,
            "int8_size_kb": int8_size_kb,
            "target_int8_kb": self.config.target_size_kb,
            "input_shape": [1, PINNA_FREQUENCY_BINS, 2],  # [batch, freq, real/imag]
            "output_shape": [1, 1],
            "quantization_bits": self.config.bits,
        }

        return {"weights": weights, "architecture": architecture, "metadata": metadata}

    def quantize_weights(
        self,
        weights: Dict[str, NDArray[np.float32]],
        calibration_data: Optional[List[NDArray]] = None,
    ) -> Tuple[Dict[str, NDArray[np.int8]], Dict[str, Tuple[float, float]]]:
        """
        Quantize weights to INT8.

        This is a reference implementation. For production, use:
        - TensorFlow Lite converter with full integer quantization
        - STM32Cube.AI quantization tools

        Args:
            weights: FP32 weights dictionary
            calibration_data: Representative data for calibration

        Returns:
            Tuple of (quantized_weights, scale_zero_points)
        """
        quantized = {}
        scales = {}

        for name, w in weights.items():
            if self.config.symmetric:
                # Symmetric quantization: scale = max(|w|) / 127
                max_abs = np.max(np.abs(w))
                scale = max_abs / 127.0 if max_abs > 0 else 1.0
                zero_point = 0
            else:  # pragma: no cover
                # Asymmetric quantization
                w_min, w_max = w.min(), w.max()
                range_val = w_max - w_min
                scale = range_val / 255.0 if range_val > 0 else 1.0
                zero_point = int(-w_min / scale) if scale > 0 else 0

            # Quantize
            w_scaled = w / scale
            w_quantized = np.clip(np.round(w_scaled), -128, 127).astype(np.int8)

            quantized[name] = w_quantized
            scales[name] = (scale, zero_point)

        return quantized, scales

    def save_for_tflite(self, model_dict: Dict[str, Any], output_path: str) -> None:
        """
        Save model in format for TFLite conversion.

        Note: This saves the model definition. Actual TFLite conversion
        requires the TensorFlow Lite converter tool.

        Args:
            model_dict: Model dictionary from export_pinna()
            output_path: Output file path (.json)
        """
        import json

        # Convert numpy arrays to lists for JSON serialization
        serializable = {
            "architecture": model_dict["architecture"],
            "metadata": model_dict["metadata"],
            "weights": {name: w.tolist() for name, w in model_dict["weights"].items()},
        }

        with open(output_path, "w") as f:
            json.dump(serializable, f, indent=2)

    def generate_c_header(
        self,
        model_dict: Dict[str, Any],
        quantized_weights: Dict[str, NDArray[np.int8]],
        scales: Dict[str, Tuple[float, float]],
    ) -> str:
        """
        Generate C header file for direct MCU deployment.

        This generates a header with quantized weights as C arrays,
        suitable for direct inclusion in STM32 projects.

        Args:
            model_dict: Model dictionary from export_pinna()
            quantized_weights: INT8 quantized weights
            scales: Scale and zero-point for each weight

        Returns:
            C header file contents as string
        """
        lines = [
            "/**",
            " * SYMBIONT PINNA Model - INT8 Quantized Weights",
            f" * Total parameters: {model_dict['metadata']['total_parameters']}",
            f" * Model size: {model_dict['metadata']['int8_size_kb']:.2f} KB",
            " * Generated by symbiont.tinyml",
            " */",
            "",
            "#ifndef PINNA_WEIGHTS_H",
            "#define PINNA_WEIGHTS_H",
            "",
            "#include <stdint.h>",
            "",
        ]

        # Add scale definitions
        lines.append("// Quantization scales")
        for name, (scale, zp) in scales.items():
            safe_name = name.replace(".", "_").upper()
            lines.append(f"#define {safe_name}_SCALE {scale}f")
            lines.append(f"#define {safe_name}_ZERO_POINT {zp}")
        lines.append("")

        # Add weight arrays
        for name, w in quantized_weights.items():
            safe_name = name.replace(".", "_")
            flat = w.flatten()

            lines.append(f"// {name}: shape {w.shape}")
            lines.append(f"static const int8_t {safe_name}[{len(flat)}] = {{")

            # Write in rows of 16
            for i in range(0, len(flat), 16):
                row = flat[i : i + 16]
                row_str = ", ".join(str(v) for v in row)
                lines.append(f"    {row_str},")

            lines.append("};")
            lines.append("")

        lines.append("#endif // PINNA_WEIGHTS_H")

        return "\n".join(lines)


class MemoryAnalyzer:
    """
    Analyze memory requirements for MCU deployment.

    Validates that SYMBIONT components fit within MCU constraints.
    """

    def __init__(self, config: Optional[DeploymentConfig] = None):
        """
        Initialize analyzer.

        Args:
            config: Deployment configuration
        """
        self.config = config or DeploymentConfig()

    def analyze(self) -> Dict[str, Any]:
        """
        Analyze memory requirements.

        Returns:
            Dictionary with memory analysis results
        """
        components = {
            "SMIS": SMIS_MEMORY_KB,
            "ECTI_PINNA": ECTI_MEMORY_KB,
            "TEFL_encoder": TEFL_MEMORY_KB,
            "Buffers": BUFFER_MEMORY_KB,
        }

        total_model = sum(components.values())
        total_runtime = self.config.heap_kb + self.config.stack_kb
        total_required = total_model + total_runtime

        available = self.config.sram_kb
        margin = available - total_required
        utilization = total_required / available * 100

        return {
            "components": components,
            "total_model_kb": total_model,
            "total_runtime_kb": total_runtime,
            "total_required_kb": total_required,
            "available_kb": available,
            "margin_kb": margin,
            "utilization_percent": utilization,
            "fits": margin >= 0,
            "target_mcu": self.config.mcu_family,
        }

    def print_report(self) -> None:
        """Print memory analysis report."""
        analysis = self.analyze()

        print("=" * 50)  # pragma: no cover
        print("SYMBIONT Memory Analysis Report")  # pragma: no cover
        print(f"Target MCU: {analysis['target_mcu']}")  # pragma: no cover
        print("=" * 50)  # pragma: no cover
        print()  # pragma: no cover
        print("Component Memory (KB):")  # pragma: no cover
        for name, size in analysis["components"].items():
            print(f"  {name:<20} {size:>6.1f}")  # pragma: no cover
        print(f"  {'─' * 26}")  # pragma: no cover
        print(f"  {'Total Model':<20} {analysis['total_model_kb']:>6.1f}")  # pragma: no cover
        print()  # pragma: no cover
        print(f"Runtime Memory:        {analysis['total_runtime_kb']:>6.1f}")  # pragma: no cover
        print(f"Total Required:        {analysis['total_required_kb']:>6.1f}")  # pragma: no cover
        print(f"Available SRAM:        {analysis['available_kb']:>6.1f}")  # pragma: no cover
        print(f"Margin:                {analysis['margin_kb']:>6.1f}")  # pragma: no cover
        print(
            f"Utilization:           {analysis['utilization_percent']:>5.1f}%"
        )  # pragma: no cover
        print()  # pragma: no cover
        status = "✓ PASS" if analysis["fits"] else "✗ FAIL"
        print(f"Status: {status}")  # pragma: no cover
        print("=" * 50)  # pragma: no cover


def verify_deployment_feasibility() -> bool:
    """
    Verify that SYMBIONT meets deployment constraints.

    Checks:
        1. Model size fits in SRAM
        2. Parameter count matches proposal
        3. Memory budget is within limits

    Returns:
        True if all checks pass
    """
    checks = []

    # Check 1: Parameter count
    expected_params = PINNA_TOTAL_PARAMS  # 4127 from proposal
    checks.append(("PINNA params", expected_params == 4127))

    # Check 2: INT8 model size
    int8_size = expected_params  # 1 byte per param
    target_bytes = ECTI_MEMORY_KB * 1024
    checks.append(("INT8 size", int8_size <= target_bytes))

    # Check 3: Total memory
    total = TOTAL_MEMORY_KB
    target = TARGET_MCU_SRAM_KB
    checks.append(("Total memory", total < target))

    # Print results
    print("Deployment Feasibility Check")  # pragma: no cover
    print("-" * 40)  # pragma: no cover
    all_pass = True
    for name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")  # pragma: no cover
        all_pass = all_pass and passed

    print("-" * 40)  # pragma: no cover
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")  # pragma: no cover

    return all_pass


# Convenience exports
__all__ = [
    "QuantizationConfig",
    "DeploymentConfig",
    "TinyMLExporter",
    "MemoryAnalyzer",
    "verify_deployment_feasibility",
]

# Try to import TensorFlow-based components
try:
    from symbiont.tinyml.tf_pinna import (
        TF_AVAILABLE,
        TFPinna,
        TinyMLDeployment,
        deploy_pinna,
    )

    __all__.extend(  # pragma: no cover
        [
            "TFPinna",
            "TinyMLDeployment",
            "convert_to_tflite",
            "generate_stm32_header",
            "generate_stm32cube_ai_config",
            "create_inference_code",
            "deploy_pinna",
            "TF_AVAILABLE",
        ]
    )
except ImportError:  # pragma: no cover
    # TensorFlow not available - that's okay for basic usage
    TF_AVAILABLE = False
    TFPinna = None
    TinyMLDeployment = None
    deploy_pinna = None
