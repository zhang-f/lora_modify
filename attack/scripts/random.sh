export PYTHONPATH=../:$PYTHONPATH
# lora/victim/$VIC_DATASET-$VIC_MODEL \
# --out_dir lora/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random10 \

for VIC_MODEL in alexnet
do
for VIC_DATASET in TinyImageNet200
do

python knockoff/adversary/transfer.py random \
models/victim/$VIC_DATASET-$VIC_MODEL \
--out_dir models/adversary/victim[$VIC_DATASET-$VIC_MODEL]-random10 \
--budget 50000 \
--queryset $VIC_DATASET \
--batch_size 32 \
-d 1 &

done
done