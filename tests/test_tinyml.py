"""
Tests for TinyML deployment module.

Tests verify:
1. Model export functionality
2. Quantization utilities
3. Memory analysis
4. Deployment feasibility checks
5. TensorFlow PINNA (if TensorFlow available)
"""

import json

import numpy as np
import pytest

from symbiont.core.constants import (
    ECTI_MEMORY_KB,
    PINNA_TOTAL_PARAMS,
    TARGET_MCU_SRAM_KB,
    TOTAL_MEMORY_KB,
)
from symbiont.tinyml import (
    DeploymentConfig,
    MemoryAnalyzer,
    QuantizationConfig,
    TinyMLExporter,
    verify_deployment_feasibility,
)

# Check if TensorFlow is available
try:
    import tensorflow  # noqa: F401

    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class TestQuantizationConfig:
    """Tests for QuantizationConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = QuantizationConfig()
        assert config.bits == 8
        assert config.symmetric is True
        assert config.target_size_kb == 4.1

    def test_custom_values(self):
        """Test custom configuration."""
        config = QuantizationConfig(bits=4, symmetric=False)
        assert config.bits == 4
        assert config.symmetric is False


class TestDeploymentConfig:
    """Tests for DeploymentConfig."""

    def test_default_mcu(self):
        """Test default MCU configuration."""
        config = DeploymentConfig()
        assert config.mcu_family == "STM32L4"
        assert config.clock_mhz == 48
        assert config.sram_kb == 32

    def test_custom_mcu(self):
        """Test custom MCU configuration."""
        config = DeploymentConfig(mcu_family="STM32F4", clock_mhz=168, sram_kb=128)
        assert config.mcu_family == "STM32F4"
        assert config.clock_mhz == 168
        assert config.sram_kb == 128


class TestTinyMLExporter:
    """Tests for TinyMLExporter."""

    @pytest.fixture
    def exporter(self):
        """Create exporter instance."""
        return TinyMLExporter()

    @pytest.fixture
    def mock_pinna(self):
        """Create mock PINNA model with weights."""

        class MockPINNA:
            def __init__(self):
                self.W_conv_real = np.random.randn(8, 5).astype(np.float32)
                self.W_conv_imag = np.random.randn(8, 5).astype(np.float32)
                self.W_recip = np.random.randn(16, 8).astype(np.float32)
                self.b_recip = np.random.randn(16).astype(np.float32)
                self.W_attn = np.random.randn(16, 16).astype(np.float32)
                self.v_attn = np.random.randn(16).astype(np.float32)
                self.W_embed = np.random.randn(16, 16).astype(np.float32)
                self.W_out = np.random.randn(1, 16).astype(np.float32)
                self.b_out = np.random.randn(1).astype(np.float32)

        return MockPINNA()

    def test_export_pinna(self, exporter, mock_pinna):
        """Test PINNA model export."""
        model_dict = exporter.export_pinna(mock_pinna)

        assert "weights" in model_dict
        assert "architecture" in model_dict
        assert "metadata" in model_dict

        # Check weights exported
        assert "conv_real" in model_dict["weights"]
        assert "conv_imag" in model_dict["weights"]
        assert "recip_W" in model_dict["weights"]
        assert "embed_W" in model_dict["weights"]
        assert "out_W" in model_dict["weights"]

    def test_metadata_content(self, exporter, mock_pinna):
        """Test metadata completeness."""
        model_dict = exporter.export_pinna(mock_pinna)
        metadata = model_dict["metadata"]

        assert "total_parameters" in metadata
        assert "fp32_size_kb" in metadata
        assert "int8_size_kb" in metadata
        assert "input_shape" in metadata
        assert "output_shape" in metadata
        assert metadata["framework"] == "SYMBIONT"
        assert metadata["model_type"] == "PINNA"

    def test_quantize_weights_symmetric(self, exporter):
        """Test symmetric INT8 quantization."""
        weights = {"test": np.array([[-1.0, 0.0, 1.0], [0.5, -0.5, 0.0]], dtype=np.float32)}

        quantized, scales = exporter.quantize_weights(weights)

        assert "test" in quantized
        assert quantized["test"].dtype == np.int8
        assert "test" in scales

        # Check quantized values are in valid range
        assert np.all(quantized["test"] >= -128)
        assert np.all(quantized["test"] <= 127)

    def test_quantize_weights_preserves_structure(self, exporter):
        """Test that quantization preserves weight structure."""
        weights = {
            "w1": np.random.randn(4, 5).astype(np.float32),
            "w2": np.random.randn(3, 3).astype(np.float32),
        }

        quantized, _ = exporter.quantize_weights(weights)

        assert quantized["w1"].shape == (4, 5)
        assert quantized["w2"].shape == (3, 3)

    def test_save_for_tflite(self, exporter, mock_pinna, tmp_path):
        """Test saving model for TFLite conversion."""
        model_dict = exporter.export_pinna(mock_pinna)
        output_path = tmp_path / "test_model.json"

        exporter.save_for_tflite(model_dict, str(output_path))

        assert output_path.exists()

        # Verify JSON is valid
        with open(output_path, "r") as f:
            loaded = json.load(f)

        assert "architecture" in loaded
        assert "metadata" in loaded
        assert "weights" in loaded

    def test_generate_c_header(self, exporter, mock_pinna):
        """Test C header generation."""
        model_dict = exporter.export_pinna(mock_pinna)
        quantized, scales = exporter.quantize_weights(model_dict["weights"])

        header = exporter.generate_c_header(model_dict, quantized, scales)

        assert "#ifndef PINNA_WEIGHTS_H" in header
        assert "#define PINNA_WEIGHTS_H" in header
        assert "static const int8_t" in header
        assert "_SCALE" in header


class TestMemoryAnalyzer:
    """Tests for MemoryAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return MemoryAnalyzer()

    def test_analyze_returns_dict(self, analyzer):
        """Test analysis returns complete dictionary."""
        analysis = analyzer.analyze()

        required_keys = [
            "components",
            "total_model_kb",
            "total_runtime_kb",
            "total_required_kb",
            "available_kb",
            "margin_kb",
            "utilization_percent",
            "fits",
            "target_mcu",
        ]

        for key in required_keys:
            assert key in analysis

    def test_components_include_all_modules(self, analyzer):
        """Test all SYMBIONT components are accounted for."""
        analysis = analyzer.analyze()
        components = analysis["components"]

        assert "SMIS" in components
        assert "ECTI_PINNA" in components
        assert "TEFL_encoder" in components
        assert "Buffers" in components

    def test_fits_within_sram(self, analyzer):
        """Test model fits within target SRAM."""
        analysis = analyzer.analyze()

        # From proposal: 8.1KB total, 32KB SRAM available
        assert analysis["total_required_kb"] < analysis["available_kb"]
        assert analysis["fits"] is True

    def test_utilization_reasonable(self, analyzer):
        """Test memory utilization is reasonable."""
        analysis = analyzer.analyze()

        # Should be between 20% and 80%
        assert analysis["utilization_percent"] > 20
        assert analysis["utilization_percent"] < 80

    def test_print_report_runs(self, analyzer, capsys):
        """Test print report executes without error."""
        analyzer.print_report()

        captured = capsys.readouterr()
        assert "Memory Analysis Report" in captured.out
        assert "PASS" in captured.out or "FAIL" in captured.out


class TestDeploymentFeasibility:
    """Tests for deployment feasibility verification."""

    def test_verify_returns_bool(self):
        """Test verification returns boolean."""
        result = verify_deployment_feasibility()
        assert isinstance(result, bool)

    def test_pinna_params_match_proposal(self):
        """Test PINNA parameter count matches proposal (4127)."""
        assert PINNA_TOTAL_PARAMS == 4127

    def test_ecti_memory_matches_proposal(self):
        """Test ECTI memory matches proposal (4.1KB)."""
        assert ECTI_MEMORY_KB == 4.1

    def test_total_memory_within_budget(self):
        """Test total memory is within SRAM budget."""
        assert TOTAL_MEMORY_KB < TARGET_MCU_SRAM_KB


class TestIntegrationTinyML:
    """Integration tests for TinyML workflow."""

    def test_full_export_quantize_workflow(self, tmp_path):
        """Test complete export and quantization workflow."""
        from symbiont.ecti import PINNA

        # Create real PINNA model
        pinna = PINNA(frequency_bins=64)

        # Export
        exporter = TinyMLExporter()
        model_dict = exporter.export_pinna(pinna)

        # Verify export
        assert model_dict["metadata"]["total_parameters"] > 0

        # Quantize
        quantized, scales = exporter.quantize_weights(model_dict["weights"])

        # Verify quantization
        for name, w in quantized.items():
            assert w.dtype == np.int8

        # Generate C header
        header = exporter.generate_c_header(model_dict, quantized, scales)
        assert len(header) > 100

        # Save to file
        output_path = tmp_path / "pinna_export.json"
        exporter.save_for_tflite(model_dict, str(output_path))
        assert output_path.exists()

    def test_memory_analysis_with_real_constants(self):
        """Test memory analysis uses real proposal constants."""
        analyzer = MemoryAnalyzer()
        analysis = analyzer.analyze()

        # Verify matches proposal values
        assert analysis["total_model_kb"] == 8.1  # From proposal
        assert analysis["available_kb"] == 32  # STM32L4+ SRAM


# =============================================================================
# TensorFlow PINNA Tests (only run if TensorFlow is available)
# =============================================================================


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
class TestTFPinna:
    """Tests for TensorFlow-based PINNA model."""

    def test_create_model(self):
        """Test PINNA model creation."""
        from symbiont.tinyml.tf_pinna import create_pinna_model

        model = create_pinna_model()
        assert model is not None
        assert model.name == "PINNA"

    def test_model_input_shape(self):
        """Test model has correct input shape."""
        from symbiont.tinyml.tf_pinna import PINNA_FREQUENCY_BINS, create_pinna_model

        model = create_pinna_model()
        input_shape = model.input_shape

        # Should be (None, 64, 2) for batch, freq_bins, mag/phase
        assert input_shape[1] == PINNA_FREQUENCY_BINS
        assert input_shape[2] == 2

    def test_model_output_shape(self):
        """Test model has correct output shape."""
        from symbiont.tinyml.tf_pinna import create_pinna_model

        model = create_pinna_model()
        output_shape = model.output_shape

        # Should be (None, 1) for anomaly score
        assert output_shape[-1] == 1

    def test_model_inference(self):
        """Test model can run inference."""
        from symbiont.tinyml.tf_pinna import PINNA_FREQUENCY_BINS, create_pinna_model

        model = create_pinna_model()

        # Create random input
        X = np.random.randn(1, PINNA_FREQUENCY_BINS, 2).astype(np.float32)

        # Run inference
        output = model.predict(X, verbose=0)

        # Check output
        assert output.shape == (1, 1)
        assert 0 <= output[0, 0] <= 1  # Sigmoid output

    def test_parameter_count(self):
        """Test model parameter count is reasonable."""
        from symbiont.tinyml.tf_pinna import count_parameters, create_pinna_model

        model = create_pinna_model()
        params = count_parameters(model)

        # TFLite-compatible model is smaller than full PINNA
        # Full PINNA: 4,127 params (NumPy simulation)
        # TFLite PINNA: ~500 params (simplified for MCU)
        # Both achieve same functionality with different architectures
        assert params > 100  # Should have meaningful params
        assert params < 10000  # Should be small for MCU


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
class TestTFLiteConversion:
    """Tests for TFLite conversion."""

    def test_convert_to_tflite(self, tmp_path):
        """Test TFLite conversion without quantization."""
        from symbiont.tinyml.tf_pinna import convert_to_tflite, create_pinna_model

        model = create_pinna_model()

        # Convert without full quantization
        tflite_model = convert_to_tflite(model, quantize=False)

        assert tflite_model is not None
        assert len(tflite_model) > 0

    def test_convert_with_quantization(self, tmp_path):
        """Test TFLite conversion with INT8 quantization."""
        from symbiont.tinyml.tf_pinna import (
            PINNA_FREQUENCY_BINS,
            convert_to_tflite,
            create_pinna_model,
        )

        model = create_pinna_model()

        # Create calibration data
        calibration_data = np.random.randn(100, PINNA_FREQUENCY_BINS, 2).astype(np.float32)

        # Convert with quantization
        output_path = tmp_path / "pinna_quantized.tflite"
        tflite_model = convert_to_tflite(
            model, representative_data=calibration_data, quantize=True, output_path=str(output_path)
        )

        assert tflite_model is not None
        assert output_path.exists()

        # Check size is reasonable (should be around 4KB for INT8)
        size_kb = len(tflite_model) / 1024
        assert size_kb < 10  # Should be well under 10KB


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
class TestSTM32Generation:
    """Tests for STM32 deployment file generation."""

    def test_generate_c_header(self):
        """Test C header generation."""
        from symbiont.tinyml.tf_pinna import generate_c_header

        # Create dummy model bytes
        dummy_model = bytes([0x00, 0x01, 0x02, 0x03, 0xFF])

        header = generate_c_header(dummy_model, model_name="test")

        assert "#ifndef TEST_MODEL_H" in header
        assert "const unsigned char test_model" in header
        assert "0x00" in header

    def test_generate_inference_code(self):
        """Test inference code generation."""
        from symbiont.tinyml.tf_pinna import generate_inference_code

        code = generate_inference_code("pinna")

        assert "pinna_init" in code
        assert "pinna_infer" in code


@pytest.mark.skipif(not TF_AVAILABLE, reason="TensorFlow not installed")
class TestTinyMLDeployment:
    """Tests for full deployment workflow."""

    def test_deployment_manager_creation(self, tmp_path):
        """Test deployment manager initialization."""
        from symbiont.tinyml.tf_pinna import TinyMLDeployment

        deployer = TinyMLDeployment(output_dir=str(tmp_path))
        assert deployer.output_dir.exists()

    def test_create_model_via_deployer(self, tmp_path):
        """Test model creation via deployment manager."""
        from symbiont.tinyml.tf_pinna import TinyMLDeployment

        deployer = TinyMLDeployment(output_dir=str(tmp_path))
        model = deployer.create_model()

        assert model is not None
        assert deployer.model is not None
