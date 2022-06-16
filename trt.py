import time

import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import torch

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import (invert_affine, postprocess, preprocess_video)


class EfficientDet_TRT:
    def __init__(self, engine_path='efficientdet-d0.trt', input_size=128):
        self.engine_path = engine_path
        self.input_size = input_size
        self.threshold = 0.2
        self.iou_threshold = 0.2
        self.obj_list = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', '',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            '', 'backpack', 'umbrella', '', '', 'handbag', 'tie', 'suitcase',
            'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', '', 'dining table', '', '',
            'toilet', '', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
            'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', '', 'book', 'clock', 'vase', 'scissors',
            'teddy bear', 'hair drier', 'toothbrush'
        ]

        self.ori_imgs = None
        self.framed_metas = None

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self.read_engine()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes(input_size, input_size)

    def read_engine(self):
        runtime = trt.Runtime(self.logger)
        with open(self.engine_path, 'rb') as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        return engine

    def alloc_buf(self):
        h_input_size = trt.volume(self.engine.get_binding_shape('input'))
        h_regression_size = trt.volume(
            self.engine.get_binding_shape('regression'))
        h_classification_size = trt.volume(
            self.engine.get_binding_shape('classification'))
        h_anchors_size = trt.volume(self.engine.get_binding_shape('anchors'))

        h_input_dtype = trt.nptype(self.engine.get_binding_dtype('input'))
        h_regression_dtype = trt.nptype(
            self.engine.get_binding_dtype('regression'))
        h_classification_dtype = trt.nptype(
            self.engine.get_binding_dtype('classification'))
        h_anchors_dtype = trt.nptype(self.engine.get_binding_dtype('anchors'))

        input_cpu = cuda.pagelocked_empty(h_input_size, h_input_dtype)
        regression_cpu = cuda.pagelocked_empty(h_regression_size,
                                               h_regression_dtype)
        classification_cpu = cuda.pagelocked_empty(h_classification_size,
                                                   h_classification_dtype)
        anchors_cpu = cuda.pagelocked_empty(h_anchors_size, h_anchors_dtype)

        input_gpu = cuda.mem_alloc(input_cpu.nbytes)
        regression_gpu = cuda.mem_alloc(regression_cpu.nbytes)
        classification_gpu = cuda.mem_alloc(classification_cpu.nbytes)
        anchors_gpu = cuda.mem_alloc(anchors_cpu.nbytes)

        return input_cpu, input_gpu, regression_cpu, regression_gpu, \
            classification_cpu, classification_gpu, anchors_cpu, anchors_gpu

    def preprocess(self, frame):
        ori_imgs, framed_imgs, framed_metas = preprocess_video(
            frame, max_size=self.input_size)
        self.ori_imgs = ori_imgs
        self.framed_metas = framed_metas
        x = np.expand_dims(framed_imgs[0], 0)
        inputs = x.transpose(0, 3, 1, 2)
        return inputs

    def execute(self, x, input_gpu, regression_cpu, regression_gpu,
                classification_cpu, classification_gpu, anchors_cpu,
                anchors_gpu):
        with self.engine.create_execution_context() as context:
            cuda.memcpy_htod(input_gpu, x)

            context.execute(1, [
                int(input_gpu),
                int(regression_gpu),
                int(classification_gpu),
                int(anchors_gpu)
            ])

            cuda.memcpy_dtoh(regression_cpu, regression_gpu)
            cuda.memcpy_dtoh(classification_cpu, classification_gpu)
            cuda.memcpy_dtoh(anchors_cpu, anchors_gpu)
        return regression_cpu, classification_cpu, anchors_cpu

    def postprocess(self, regression, classification, anchors):
        regression = regression.reshape(1, 3069, 4)
        classification = classification.reshape(1, 3069, 90)
        anchors = anchors.reshape(1, 3069, 4)

        regression = torch.from_numpy(regression)
        classification = torch.from_numpy(classification)
        anchors = torch.from_numpy(anchors)

        out = postprocess(1, regression, classification, anchors,
                          self.regressBoxes, self.clipBoxes, self.threshold,
                          self.iou_threshold)
        out = invert_affine(self.framed_metas, out)
        return out

    def infer(self, frame):
        t1 = time.time()
        # preprocess images
        inputs = self.preprocess(frame)

        t2 = time.time()
        # allocate buffers
        input_cpu, input_gpu, regression_cpu, regression_gpu, \
            classification_cpu, classification_gpu, anchors_cpu, \
            anchors_gpu = self.alloc_buf()

        # run through trt engine
        regression, classification, anchors = self.execute(
            inputs.reshape(-1), input_gpu, regression_cpu, regression_gpu,
            classification_cpu, classification_gpu, anchors_cpu, anchors_gpu)

        t3 = time.time()
        # postprocess results
        out = self.postprocess(regression, classification, anchors)

        t4 = time.time()
        print(f"preprocess: {(t2-t1)*1000:.2f} ms, infer: {(t3-t2)*1000:.2f}"
              f" ms, postproc.: {(t4-t3)*1000:.2f} ms")
        return out

    def display(self, preds, imgs):
        for i in range(len(imgs)):
            if len(preds[i]['rois']) == 0:
                return imgs[i]

            for j in range(len(preds[i]['rois'])):
                score = float(preds[i]['scores'][j])
                if preds[i]['class_ids'][j] > 3:  # score < 0.5:
                    continue

                (x1, y1, x2, y2) = preds[i]['rois'][j].astype(np.int)
                cv2.rectangle(imgs[i], (x1, y1), (x2, y2), (0, 255, 255), 2)
                obj = self.obj_list[preds[i]['class_ids'][j]]

                cv2.putText(imgs[i], '{}, {:.3f}'.format(obj,
                                                         score), (x1, y1 + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            return imgs[i]

    def annotate(self, out):
        img_show = self.display(out, self.ori_imgs)
        return img_show


if __name__ == "__main__":
    model = EfficientDet_TRT()

    video_src = './nyc.mp4'
    cap = cv2.VideoCapture(video_src)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, 1000)
    ind = 0

    while True:
        ind += 1
        ret, frame = cap.read()
        if not ret:
            break

        out = model.infer(frame)
        img_show = model.annotate(out)

        # show frame by frame
        cv2.imshow('frame', img_show)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == 32:
            cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
