#!/bin/sh

CUDA=7
export CUDA_VISIBLE_DEVICES=$CUDA
for exp in 1 2 3
do
	for SEED in 1 2 3
	do
		python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 300 --gamma 0.1 --dog_group ${exp} --lisa_mix_up --mix_alpha 2 --cut_mix --group_by_label --root_dir /data1/taero/data/MetaDatasetCatDog/ --log_dir ./logs/meta_exp${exp}/lisa_seed_${SEED} --seed ${SEED}
	done
done
