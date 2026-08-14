import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
import tensorrt as trt
import os
import onnx

# ================================================================
# CONFIG (ARGUMENTS)
# ================================================================
NUM_CLASSES       = 6
CLASSIFIER_PTH    = "classifier_weights/best_model.pth"
CLASSIFIER_ONNX   = "classifier_weights/classifier.onnx"
CLASSIFIER_ENGINE = "classifier_weights/classifier_fp16.engine"

INPUT_SHAPE       = (1, 3, 224, 224)
OPSET_VERSION     = 18
FP16_ENABLED      = True
WORKSPACE_SIZE    = 8 << 30  # 8GB


# ================================================================
# PART 1: PyTorch → ONNX
# ================================================================
print("📦 Exporting MobileNetV3 classifier to ONNX...")

model = mobilenet_v3_small()
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    NUM_CLASSES
)

state_dict = torch.load(CLASSIFIER_PTH, map_location="cpu")
model.load_state_dict(state_dict)
model.eval()

dummy_input = torch.randn(*INPUT_SHAPE)

torch.onnx.export(
    model,
    dummy_input,
    CLASSIFIER_ONNX,
    input_names=["input"],
    output_names=["output"],
    opset_version=OPSET_VERSION,
    export_params=True,
    do_constant_folding=True,
)

print(f"✅ ONNX exported: {CLASSIFIER_ONNX}")


# ================================================================
# FIX: Ensure NO external weight file (.onnx.data issue fix)
# ================================================================
print("🧹 Cleaning ONNX (removing external weight dependency)...")

onnx_model = onnx.load(CLASSIFIER_ONNX)
onnx.save_model(
    onnx_model,
    CLASSIFIER_ONNX,
    save_as_external_data=False
)

print("✅ ONNX cleaned and ready for TensorRT")


# ================================================================
# PART 2: TensorRT Engine Builder
# ================================================================
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_classifier_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
) -> bool:

    if not os.path.exists(onnx_path):
        print(f"❌ ONNX file not found: {onnx_path}")
        return False

    print(f"🔄 Building TensorRT engine from: {onnx_path}")

    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_SIZE)

        # FP16
        if fp16 and torch.cuda.is_available():
            major = torch.cuda.get_device_capability()[0]
            if major >= 7:
                print("⚡ FP16 enabled")
                config.set_flag(trt.BuilderFlag.FP16)

        # Parse ONNX
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                print("❌ ONNX parse failed:")
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                return False

        print("✅ ONNX parsed successfully")

        # FIX: Explicit input shape (VERY IMPORTANT for TRT 10)
        input_tensor = network.get_input(0)
        input_tensor.shape = INPUT_SHAPE

        print("⚙️ Building TensorRT engine...")

        serialized_engine = builder.build_serialized_network(network, config)

        if serialized_engine is None:
            print("❌ Engine build failed")
            return False

        os.makedirs(os.path.dirname(engine_path), exist_ok=True)

        with open(engine_path, "wb") as f:
            f.write(serialized_engine)

        print(f"🚀 TensorRT engine saved: {engine_path}")
        return True


# ================================================================
# RUN BUILD
# ================================================================
classifier_ok = build_classifier_engine(
    CLASSIFIER_ONNX,
    CLASSIFIER_ENGINE,
    fp16=FP16_ENABLED
)

print("\n" + "=" * 60)
print("BUILD SUMMARY")
print("=" * 60)
print(f"ONNX Export        : ✅ OK ({CLASSIFIER_ONNX})")
print(f"TensorRT Engine    : {'✅ OK' if classifier_ok else '❌ FAILED'} ({CLASSIFIER_ENGINE})")