#!/usr/bin bash

for s in 0
do 
    python ADB.py \
        --dataset banking \
        --known_cls_ratio 1 \
        --labeled_ratio 1 \
        --seed $s \
        --freeze_bert_parameters \
        --gpu_id 0
done