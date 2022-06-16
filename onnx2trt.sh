#!/bin/bash

# convert onnx to trt engine with trtexec
../TensorRT-7.2.3.4/bin/trtexec --onnx=./efficientdet-d0.onnx --saveEngine=efficientdet-d0.trt --workspace=1024

# speed test
../TensorRT-7.2.3.4/bin/trtexec --loadEngine=./efficientdet-d0.trt --useCudaGraph --noDataTransfers --iterations=10 --avgRuns=10
