#!/bin/sh

RHO=0.05
LR=1e-5
WD=1

SEED=0
CUDA=3
export CUDA_VISIBLE_DEVICES=$CUDA
nohup python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --root_dir /data1/taero/data/cub --lr ${LR} --batch_size 16 --weight_decay ${WD} --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 2 --reweight_groups --log_dir /data1/taero/worst_sharp/subpopulation_shifts/logs/cub_asgdro_rho_${RHO}_balance_lr_${LR}_wd_${WD}_seed_${SEED} --rho ${RHO} --asgdro --save_best --save_last --seed $SEED &

SEED=1
CUDA=5
export CUDA_VISIBLE_DEVICES=$CUDA
nohup python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --root_dir /data1/taero/data/cub --lr ${LR} --batch_size 16 --weight_decay ${WD} --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 2 --reweight_groups --log_dir /data1/taero/worst_sharp/subpopulation_shifts/logs/cub_asgdro_rho_${RHO}_balance_lr_${LR}_wd_${WD}_seed_${SEED} --rho ${RHO} --asgdro --save_best --save_last --seed $SEED &

SEED=2
CUDA=4
export CUDA_VISIBLE_DEVICES=$CUDA
nohup python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --root_dir /data1/taero/data/cub --lr ${LR} --batch_size 16 --weight_decay ${WD} --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 2 --reweight_groups --log_dir /data1/taero/worst_sharp/subpopulation_shifts/logs/cub_asgdro_rho_${RHO}_balance_lr_${LR}_wd_${WD}_seed_${SEED} --rho ${RHO} --asgdro --save_best --save_last --seed $SEED &

