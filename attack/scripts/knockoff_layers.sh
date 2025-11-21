export PYTHONPATH=../:$PYTHONPATH


# VIC_MODEL=resnet50
# VIC_DATASET=CIFAR10
# for VIC_MODEL in resnet50 resnet18
# do
# for VIC_DATASET in CIFAR10 CIFAR100
# do
for VIC_MODEL in alexnet
do
for VIC_DATASET in CIFAR100
do
for MODE in block_deep
do


python knockoff/adversary/train_layers.py \
models/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random10 \
$VIC_MODEL $VIC_DATASET \
models/victim/$VIC_DATASET-$VIC_MODEL \
--budgets 50,100,150,200,250,300,350,400,450,500 \
-d 1 \
--pretrained imagenet_for_cifar \
--log-interval 50 \
--epochs 20 \
--lr 0.05 \
--remain-lr 1e-3 \
--update-lr 1e-2 \
--graybox-mode $MODE \
--argmaxed \

done
done
done