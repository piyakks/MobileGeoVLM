#
# Quantize exported FastVLM ONNX graphs for mobile CPU:
#   decoder      -> 4-bit weight-only (MatMulNBits)
#   embed_tokens -> int8 (Gather)
#   vision       -> int8 dynamic (optional; compare quality against fp32 before using)
#
# Usage:
#   python onnx_export/quantize_onnx.py --onnx-dir onnx_export/fastvlm_0.5b
#
import os
import argparse

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer


def quantize_decoder(src, dst, block_size, accuracy_level):
    model = onnx.load(src)
    quant = MatMulNBitsQuantizer(model, block_size=block_size, is_symmetric=True, accuracy_level=accuracy_level)
    quant.process()
    quant.model.save_model_to_file(dst, use_external_data_format=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-dir", required=True)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--accuracy-level", type=int, default=4, help="4 = int8 compute (fastest on ARM)")
    args = parser.parse_args()
    d = args.onnx_dir

    print("Quantizing decoder to 4-bit ...")
    quantize_decoder(os.path.join(d, "decoder.onnx"), os.path.join(d, "decoder_q4.onnx"),
                     args.block_size, args.accuracy_level)

    print("Quantizing embed_tokens to int8 ...")
    quantize_dynamic(os.path.join(d, "embed_tokens.onnx"), os.path.join(d, "embed_tokens_int8.onnx"),
                     op_types_to_quantize=["Gather"], weight_type=QuantType.QUInt8)

    print("Quantizing vision_encoder to int8 ...")
    quantize_dynamic(os.path.join(d, "vision_encoder.onnx"), os.path.join(d, "vision_encoder_int8.onnx"),
                     op_types_to_quantize=["MatMul", "Conv"], weight_type=QuantType.QUInt8)

    for f in sorted(os.listdir(d)):
        if f.endswith((".onnx", ".data")):
            print(f"{os.path.getsize(os.path.join(d, f)) / 1e6:9.1f} MB  {f}")
