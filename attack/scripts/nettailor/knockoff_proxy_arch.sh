export PYTHONPATH=../:$PYTHONPATH

PROXY_MODEL=alexnet


for VIC_MODEL in alexnet
do
for VIC_DATASET in CIFAR10 
do
for iter in 50 150 200 250 300 350 400 450 500
do

python knockoff/adversary/train_nettailor.py \
models/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random10 \
$PROXY_MODEL $VIC_DATASET \
models/victim/$VIC_DATASET-$VIC_MODEL \
--budgets $iter \
-d 0 \
--pretrained imagenet_for_cifar \
--log-interval 100 \
--epochs 10 \
--lr 0.1 \
--backbone-lr 0.01 \
--argmaxed \
# &

done
done
done