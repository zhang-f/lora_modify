export PYTHONPATH=../:$PYTHONPATH


for VIC_MODEL in vgg19
do
for VIC_DATASET in ImageNet1k   # CIFAR10 CIFAR100 TinyImageNet200
do

python ../knockoff/adversary/transfer_lora.py random \
victim/$VIC_DATASET-$VIC_MODEL \
--out_dir models/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random10 \
--budget 50000 \
--queryset $VIC_DATASET \
--batch_size 32 \
-d 1 &

done
done