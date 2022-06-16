import torch
from backbone import EfficientDetBackbone

obj_list = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', '', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', '', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

compound_coef = 0
input_size = 512
batch_size = 1

model = EfficientDetBackbone(compound_coef=compound_coef,
                             num_classes=len(obj_list),
                             onnx_export=True)

model.load_state_dict(torch.load(f'weights/efficientdet-d{compound_coef}.pth'))
model.eval()

x = torch.randn(batch_size,
                3,
                input_size,
                input_size,
                requires_grad=False)


print('exporting...')
torch.onnx.export(
    model,
    x,
    f'efficientdet-d{compound_coef}.onnx',
    input_names=['input'],
    output_names=['regression', 'classification', 'anchors'],
    opset_version=10)
