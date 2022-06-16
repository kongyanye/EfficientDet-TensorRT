# EfficientDet-TensorRT

TensorRT speedup for EfficientDet models. The repo is based on https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch.

## Usage

Speeding up the EfficientDet models with TensorRT is mainly composed of 3 steps:

1. Modify original code to support TensorRT speedup (refer to [link](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/issues/111#issuecomment-657422226) and [link](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/issues/29#issuecomment-618904458))

2. Convert Pytorch model to onnx file: `python torch2onnx.py`

3. Convert onnx file to TensorRT engine: `bash onnx2trt.sh`

4. Infer with TensorRT engine: `python trt.py`

## Performance

For Efficientdet-D0 and input size 128 (on RTX 3090):

1. Before speedup: ~35 ms 
2. After speedup: ~9 ms

Note: The current codebase implement EfficientDet-D0 with input size 128 for TensorRT speedup. Other models or input sizes can be realized by modifying the related code. Feel free to ask questions.
