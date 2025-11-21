#!/bin/bash
export PYTHONPATH=../:$PYTHONPATH

for DATASET in CIFAR10  # CIFAR10 CIFAR100 TinyImageNet200
do
for MODEL_ARCH in vgg19
do
for SCALE_FACTOR in 0.01 # You can adjust this value as needed
do
for MAX_MODIFIED_WEIGHT_RATIO in 0.0001 # You can adjust this value as needed
do
for TOPK in 1000 5000 10000   # You can adjust this value as needed
do

TARGET_CLASS=3
BATCH_SIZE=100
DEVICE="cuda:1"  # or "cpu" if you don't have a GPU
PRETRAINED_PATH="../models/victim/$DATASET-$MODEL_ARCH/checkpoint.pth.tar" # OBFUSCATED_PATH="victim/$DATASET-$MODEL_ARCH/checkpoint.pth.tar"

python adaptive_fisher.py \
    --dataset $DATASET \
    --model_arch $MODEL_ARCH \
    --target_class $TARGET_CLASS \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --model_pretrained imagenet_for_cifar \
    --pretrained_path $PRETRAINED_PATH \
    --max_modified_weight_ratio $MAX_MODIFIED_WEIGHT_RATIO \
    --scale_factor $SCALE_FACTOR \
    --topk $TOPK


done
done
done
done
done