#!/bin/sh

CUDA=6
export CUDA_VISIBLE_DEVICES=$CUDA
for SEED in 1 2 3
do

	python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 100 --gamma 0.1 --dog_group 3 --reweight_groups --asgdro --robust --root_dir /data2/taero/data/MetaDatasetCatDog/ --log_dir ./logs/meta_exp3/default_seed_${SEED} --seed ${SEED}

done
