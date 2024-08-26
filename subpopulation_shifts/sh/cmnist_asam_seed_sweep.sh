#!/bin/sh

CUDA=3

export CUDA_VISIBLE_DEVICES=$CUDA

for RHO in 0.05 0.2
do
	for SEED in 0 1 2
	do
		python run_expt.py -s confounder -d CMNIST -t 0-4 -c isred --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 50 --gamma 0.1 --generalization_adjustment 0 --reweight_groups --asam --robust --log_dir /data1/taero/worst_sharp/subpopulation_shifts/logs/cmnist_asam/rho_${RHO}_${SEED} --root_dir /data1/taero/data/ --seed $SEED --rho $RHO
	done
done
