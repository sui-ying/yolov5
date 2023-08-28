#!/bin/bash

source_data_path="/cv/xyc/Datasets/taiguo_yancan/labeled/seg/side"
target_data_path="/cv/all_training_data/yancan/xinhaida/seg/side"

python generate_datasets.py \
  --source_path ${source_data_path} \
  --target_path ${target_data_path} \
  --RGBA2RGB True \
  --resize True \
  --resize_size 2460 720

python generate_datasets_expand.py \
  --source_path ${source_data_path} \
  --target_path ${target_data_path}