#!/bin/bash

cd ../..

# val
python3 val.py \
  --weights  /cv/xyc/code/yolov5/runs/demo/train/exp4/weights/best.pt \
  --data  /cv/xyc/code/yolov5/configs/demo/dataset.yaml \
  --task val \
  --batch-size 4 \
  --device  0  \
  --project runs/demo/val
