#!/usr/bin/env bash
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=$1
for d in 'banking'
do
	for l in 0.5
	do
        	for k in 0 0.75
		do
			for s in 0
        		do
	    		python clnn.py \
				--data_dir data \
				--dataset $d \
				--known_cls_ratio $k \
				--labeled_ratio $l \
				--seed $s \
				--lr '1e-5' \
				--save_results_path 'clnn_outputs' \
				--view_strategy 'rtr' \
				--update_per_epoch 1 \
				--topk 10 \
				--bert_model "./pretrained_models/${d}"
    			done
		done
	done 
done
