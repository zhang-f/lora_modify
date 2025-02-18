#!/bin/bash

export PYTHONPATH=../:$PYTHONPATH

#!/bin/bash

# Set the dataset and model architecture TinyImageNet200 ImageNet1k
for DATASET in CIFAR10
do
for MODEL_ARCH in resnet18
do

# Set the paths to your pretrained model and checkpoint
PRETRAINED_PATH="../models/victim/$DATASET-$MODEL_ARCH/checkpoint.pth.tar"
PARAMS_JSON_PATH="../models/victim/$DATASET-$MODEL_ARCH/params.json"
# Set the target class and other parameters
TARGET_CLASS=3  # Example: Class 3
MAX_MODIFIED_WEIGHT_RATIO=0.00003
SCALE_FACTOR=0.002
BATCH_SIZE=100
DEVICE="cuda"  # or "cpu" if no GPU available

# Run the Python script with the arguments
python mask_model.py \
  --dataset $DATASET \
  --model_arch $MODEL_ARCH \
  --target_class $TARGET_CLASS \
  --max_modified_weight_ratio $MAX_MODIFIED_WEIGHT_RATIO \
  --scale_factor $SCALE_FACTOR \
  --batch_size $BATCH_SIZE \
  --device $DEVICE \
  --fisher False \
  --model_pretrained imagenet_for_cifar \
  --pretrained_path $PRETRAINED_PATH

# You can add any additional commands below if needed (like logging)

done
done
