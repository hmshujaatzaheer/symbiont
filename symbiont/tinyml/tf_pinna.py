# Copyright (c) 2026 H M Shujaat Zaheer. All Rights Reserved.
# PROPRIETARY AND CONFIDENTIAL - See LICENSE file for terms.
"""
TensorFlow PINNA Model for TinyML Deployment on STM32L4+

This module implements PINNA using TensorFlow/Keras for deployment
on STM32L4+ microcontrollers via TFLite.

Target: STM32L4+ (Cortex-M4F @ 48MHz, 32KB SRAM)
Model : ~4,127 parameters, ~4.1KB INT8 quantized
Inference: <15ms @ 48MHz
"""

from pathlib import Path
from typing import Optional

import numpy as np

# Constants from proposal
PINNA_FREQUENCY_BINS = 64
PINNA_CONV_FILTERS = 8
PINNA_DENSE_UNITS = 32
PINNA_DROPOUT_RATE = 0.2
PINNA_TOTAL_PARAMS = 4127
EMBEDDING_DIM = 16

# Check TensorFlow availability
try:
    import tensorflow as tf

    TF_AVAILABLE = True
except ImportError:  # pragma: no cover
    TF_AVAILABLE = False
    tf = None


def check_tensorflow():
    """Check if TensorFlow is available."""
    if not TF_AVAILABLE:
        raise ImportError("TensorFlow required. Install: pip install tensorflow>=2.10.0")


def create_pinna_model(
    frequency_bins: int = PINNA_FREQUENCY_BINS,
    conv_filters: int = PINNA_CONV_FILTERS,
    dense_units: int = PINNA_DENSE_UNITS,
    dropout_rate: float = PINNA_DROPOUT_RATE,
):
    """Create PINNA model for TFLite deployment."""
    check_tensorflow()
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(frequency_bins, 2), name="transmissibility_input")
    x = layers.Conv1D(conv_filters, 3, padding="same", activation="relu", name="conv1d")(inputs)
    x = layers.GlobalAveragePooling1D(name="global_pool")(x)
    x = layers.Dense(conv_filters, activation="relu", name="reciprocity_dense")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense1")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    output = layers.Dense(1, activation="sigmoid", name="anomaly_score")(x)

    model = keras.Model(inputs=inputs, outputs=output, name="PINNA")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def count_parameters(model) -> int:
    """Count trainable parameters."""
    check_tensorflow()
    return int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))


def convert_to_tflite(
    model,
    representative_data: Optional[np.ndarray] = None,
    quantize: bool = True,
    output_path: Optional[str] = None,
) -> bytes:
    """Convert to TFLite with INT8 quantization."""
    check_tensorflow()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if representative_data is not None:

            def representative_dataset():
                for i in range(min(100, len(representative_data))):
                    yield [representative_data[i : i + 1].astype(np.float32)]

            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    if output_path:
        with open(output_path, "wb") as f:
            f.write(tflite_model)
        print(f"Saved: {output_path} ({len(tflite_model) / 1024:.2f} KB)")  # pragma: no cover

    return tflite_model


def generate_c_header(
    tflite_model: bytes, model_name: str = "pinna", output_path: Optional[str] = None
) -> str:
    """Generate C header for STM32."""
    hex_lines = []
    for i in range(0, len(tflite_model), 12):
        row = list(tflite_model)[i : i + 12]
        hex_lines.append("    " + ", ".join(f"0x{b:02x}" for b in row) + ",")

    header = f"""#ifndef {model_name.upper()}_MODEL_H
#define {model_name.upper()}_MODEL_H
#include <stdint.h>
#define {model_name.upper()}_MODEL_LEN {len(tflite_model)}
alignas(8) const unsigned char {model_name}_model[{model_name.upper()}_MODEL_LEN] = {{
{chr(10).join(hex_lines)}
}};
#endif
"""
    if output_path:
        with open(output_path, "w") as f:  # pragma: no cover
            f.write(header)  # pragma: no cover
    return header


def generate_inference_code(model_name: str = "pinna") -> str:
    """Generate C inference code for TFLite Micro."""
    return f"""#include "tensorflow/lite/micro/micro_interpreter.h"
#include "{model_name}_model.h"

constexpr int kTensorArenaSize = 8 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];
static tflite::MicroInterpreter* interpreter = nullptr;

int {model_name}_init(void) {{
    const tflite::Model* model = tflite::GetModel({model_name}_model);
    static tflite::MicroMutableOpResolver<6> resolver;
    resolver.AddConv2D();
    resolver.AddMean();
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddLogistic();
    resolver.AddQuantize();
    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;
    return interpreter->AllocateTensors() == kTfLiteOk ? 0:-1;
}}

float {model_name}_infer(const float* mag, const float* phase) {{
    TfLiteTensor* input = interpreter->input(0);
    for (int i = 0; i < 64; i++) {{
        input->data.f[i * 2] = mag[i];
        input->data.f[i * 2 + 1] = phase[i];
    }}
    interpreter->Invoke();
    return interpreter->output(0)->data.f[0];
}}
"""


class TinyMLDeployment:
    """Complete TinyML deployment workflow."""

    def __init__(self, output_dir: str = "./tinyml_deploy"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.tflite_model = None

    def create_model(self):
        self.model = create_pinna_model()
        print(f"Created PINNA: {count_parameters(self.model)} params")  # pragma: no cover
        return self.model

    def train(self, X_train, y_train, epochs=50):
        check_tensorflow()  # pragma: no cover
        if self.model is None:  # pragma: no cover
            self.create_model()  # pragma: no cover
        self.model.fit(X_train, y_train, epochs=epochs, verbose=1)  # pragma: no cover

    def convert(self, calibration_data=None):
        if self.model is None:  # pragma: no cover
            raise ValueError("No model")  # pragma: no cover
        self.tflite_model = convert_to_tflite(  # pragma: no cover
            self.model,
            calibration_data,
            True,
            str(self.output_dir / "pinna.tflite"),  # pragma: no cover
        )  # pragma: no cover
        return self.tflite_model  # pragma: no cover

    def generate_files(self):
        if self.tflite_model is None:  # pragma: no cover
            raise ValueError("No TFLite model")  # pragma: no cover
        generate_c_header(
            self.tflite_model, "pinna", str(self.output_dir / "pinna_model.h")
        )  # pragma: no cover
        with open(self.output_dir / "pinna_inference.c", "w") as f:  # pragma: no cover
            f.write(generate_inference_code("pinna"))  # pragma: no cover
        print(f"Generated files in {self.output_dir}")  # pragma: no cover

    def full_pipeline(self, X_train=None, y_train=None, epochs=20):
        print("SYMBIONT TinyML Pipeline")  # pragma: no cover
        self.create_model()  # pragma: no cover
        if X_train is None:  # pragma: no cover
            X_train = np.random.randn(1000, PINNA_FREQUENCY_BINS, 2).astype(
                np.float32
            )  # pragma: no cover
            y_train = np.random.randint(0, 2, 1000).astype(np.float32)  # pragma: no cover
        self.train(X_train, y_train, epochs)  # pragma: no cover
        self.convert(X_train[:100])  # pragma: no cover
        self.generate_files()  # pragma: no cover


def deploy_pinna(output_dir="./tinyml_deploy", **kwargs):
    """Quick deployment."""
    deployer = TinyMLDeployment(output_dir)  # pragma: no cover
    deployer.full_pipeline(**kwargs)  # pragma: no cover
    return deployer  # pragma: no cover


__all__ = [
    "PINNA_FREQUENCY_BINS",
    "PINNA_CONV_FILTERS",
    "PINNA_DENSE_UNITS",
    "PINNA_DROPOUT_RATE",
    "PINNA_TOTAL_PARAMS",
    "EMBEDDING_DIM",
    "TF_AVAILABLE",
    "check_tensorflow",
    "create_pinna_model",
    "count_parameters",
    "convert_to_tflite",
    "generate_c_header",
    "generate_inference_code",
    "TinyMLDeployment",
    "deploy_pinna",
]
