export PYTHONPATH=../:$PYTHONPATH


# VIC_MODEL=resnet50
# VIC_DATASET=CIFAR10
# for VIC_MODEL in resnet50 resnet18
# do
# for VIC_DATASET in CIFAR10 CIFAR100
# do
PROXY_MODEL=vgg19
for VIC_MODEL in vgg19
do
for VIC_DATASET in CIFAR10
do


python knockoff/adversary/eval_stealing_nettailor.py \
models/adversary/victim[CIFAR10-resnet18]-random10 \
$PROXY_MODEL $VIC_DATASET \
models/victim/$VIC_DATASET-$VIC_MODEL \
--budgets 500 \
--pretrained imagenet_for_cifar \
-d 1 \
# &


done
done