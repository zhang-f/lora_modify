export PYTHONPATH=../:$PYTHONPATH


# VIC_MODEL=resnet50
# VIC_DATASET=CIFAR10
# for VIC_MODEL in resnet50 resnet18
# do
# for VIC_DATASET in CIFAR10 CIFAR100
# do
for VIC_MODEL in vgg19
do
for VIC_DATASET in TinyImageNet200   # CIFAR10 CIFAR100 TinyImageNet200
do



python knockoff/adversary/train_nnsplitter.py \
models/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random10 \
$VIC_MODEL $VIC_DATASET \
lora/victim/$VIC_DATASET-$VIC_MODEL \
--budgets 5000 \
-d 0 \
--pretrained imagenet \
--resume models/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random10 \
--log-interval 100 \
--epochs 10 \
--lr 0.01 \
--remain-lr 1e-3 \
--update-lr 1e-2 \
--argmaxed \
&

done
done