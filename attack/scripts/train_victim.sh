export PYTHONPATH=../:$PYTHONPATH

# MODEL=resnet18, alexnet ( --lr 0.1 \)
# MODEL=vgg19 ( --lr 0.01\)
for MODEL in alexnet
do
for DATASET in TinyImageNet200
do

python knockoff/victim/train.py $DATASET $MODEL \
-d 1 \
-o models/victim/$DATASET-$MODEL \
-e 50 \
--log-interval 25 \
--pretrained imagenet \
--lr 0.01 \
--lr-step 10 


done
done