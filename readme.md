# EfficientDet-TensorRT

TensorRT speedup for EfficientDet models. The repo is based on https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch.

## Usage

Speeding up the EfficientDet models with TensorRT is mainly composed of 5 steps:

1. Modify original code to support TensorRT speedup (refer to [link](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/issues/111#issuecomment-657422226) and [link](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/issues/29#issuecomment-618904458))

2. Convert Pytorch model to onnx file: `python torch2onnx.py`

3. Visualize the onnx file with [netron](https://github.com/lutzroeder/netron): `netron efficientdet-d0.onnx`

4. Convert onnx file to TensorRT engine: `bash onnx2trt.sh`

5. Infer with TensorRT engine: `python trt.py`

## Performance

On a single RTX 3090 GPU:

| model | Input size | Inference Latency (before) | Inference Latency (after) |
| :-----: | :-----: | :------: | :------: |
| D0 | 128 | 35 ms | 9 ms |
| D0 | 512 | 39 ms | 25 ms |

Note: The current codebase implement EfficientDet-D0 with input size 512 for TensorRT speedup. Other models or input sizes can be realized by modifying the related code. Feel free to ask questions.
