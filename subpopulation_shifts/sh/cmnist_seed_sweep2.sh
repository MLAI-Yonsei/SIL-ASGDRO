#!/bin/sh

CUDA=4

export CUDA_VISIBLE_DEVICES=$CUDA

for RHO in 0.5 0.8
do
	for SEED in 0 1 2
	do
		python run_expt.py -s confounder -d CMNIST -t 0-4 -c isred --lr 0.001 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 50  --gamma 0.1 --generalization_adjustment 0 --reweight_groups --asgdro --robust --log_dir /data1/taero/worst_sharp/subpopulation_shifts/logs/cmnist/rho_${RHO}_${SEED} --root_dir /data1/taero/data/ --seed $SEED --rho $RHO
	done
done
