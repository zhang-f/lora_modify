export PYTHONPATH=../:$PYTHONPATH
#!/bin/bash

# Set the dataset and model architecture parameters
for DATASET in TinyImageNet200 # CIFAR10 CIFAR100 TinyImageNet200
do
for MODEL_ARCH in alexnet
do

TARGET_CLASS=3
MAX_MODIFIED_WEIGHT_RATIO=0.00005
SCALE_FACTOR=0.002
BATCH_SIZE=100
DEVICE="cuda"  # or "cpu" if you don't have a GPU
PRETRAINED_PATH="victim/$DATASET-$MODEL_ARCH/checkpoint.pth.tar"

# Run the Python script
python clip_debug.py \
    --dataset $DATASET \
    --model_arch $MODEL_ARCH \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --model_pretrained imagenet \
    --pretrained_path $PRETRAINED_PATH

done
done
