export PYTHONPATH=../:$PYTHONPATH
#!/bin/bash

# Set the dataset and model architecture parameters
for DATASET in ImageNet1k   # CIFAR10 CIFAR100 TinyImageNet200
do
for MODEL_ARCH in vgg19
do
for LRANK in 16
do
TARGET_CLASS=3
MAX_MODIFIED_WEIGHT_RATIO=0.0001
SCALE_FACTOR=0.001
BATCH_SIZE=100
DEVICE="cuda"  # or "cpu" if you don't have a GPU
PRETRAINED_PATH="victim/$DATASET-$MODEL_ARCH/checkpoint.pth.tar"

# Run the Python script
python lora_model.py \
    --dataset $DATASET \
    --model_arch $MODEL_ARCH \
    --target_class $TARGET_CLASS \
    --max_modified_weight_ratio $MAX_MODIFIED_WEIGHT_RATIO \
    --scale_factor $SCALE_FACTOR \
    --batch_size $BATCH_SIZE \
    --rank $LRANK \
    --device $DEVICE \
    --model_pretrained imagenet \
    --pretrained_path $PRETRAINED_PATH

done
done
done