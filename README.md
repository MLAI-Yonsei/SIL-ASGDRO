# Sufficient Invariant Learning for Distribution Shift

The code implementation for the paper, Sufficient Invariant Learning for Distribution Shift.
The code is based on [LISA](https://github.com/huaxiuyao/LISA) which is also based on the code in [groupDRO](https://github.com/kohpangwei/group_DRO) and [fish](https://github.com/YugeTen/fish).


## Prerequisites
- python 3.8.16
- matplotlib 3.7.1
- numpy 1.24.4
- pandas 1.5.3
- pillow 9.5.0
- pytorch 1.13.1
- pytorch_transformers 1.2.0
- torchvision 0.14.1
- torchaudio 0.13.1
- tqdm 4.65.5
- wilds 2.0.0
- transformers 4.27.4
- ipdb 0.13.13 
- pexpect 4.9.0
- traitlets 5.14.3
- psutil 6.0.0
- torch_scatter 

## Subpopulation Shifts and MetaShifts

To run the code, you need to first enter the directory: `cd subpopulation_shifts`. Then change the `root_dir` variable in `./data/data.py` if you need to put the dataset elsewhere other than `./data/`. 

The command for experiments are as follows:



For subpopulation shifts problems, the datasets are listed as follows:


#### CMNIST
This dataset is constructed from MNIST. It will be automatically downloaded.

```
python run_expt.py -s confounder -d CMNIST -t 0-4 -c isred --lr 0.001 --rho 0.8 --batch_size 16 --weight_decay 0.0001 --model resnet50 --n_epochs 50  --gamma 0.1 --generalization_adjustment 0 --reweight_groups --asgdro --robust --log_dir /my/log/path --root_dir /my/dataset/path/ --seed 0
```

#### CelebA
This dataset can be downloaded via the link in the repo [group_DRO](https://github.com/kohpangwei/group_DRO). 

```
python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.0001 --batch_size 16 --weight_decay 0.1 --rho --model resnet50 --n_epochs 50 --gamma 0.1 --generalization_adjustment 1 --root_dir /my/dataset/path/celebA --log_dir /my/log/path --reweight_groups --robust --asgdro --save_best --seed 0
```

#### Waterbirds
This dataset can be downloaded via the link in the repo [group_DRO](https://github.com/kohpangwei/group_DRO). 

```
python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --root_dir /my/dataset/path/cub --lr 1e-5 --batch_size 16 --weight_decay 1.0 --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 2 --reweight_groups --log_dir /my/log/path --rho 0.05 --asgdro --save_best --seed 0
```

#### MetaShifts
The dataset can be downloaded [[here]](https://drive.google.com/file/d/1Fr2HxUOL3_QUDHU5B3MMH7dgFu_u_gJ_/view?usp=sharing). You should put it under the directory `data`. 

The running scripts for 4 dataset with different distances are as follows:
```
python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --rho 0.5 --model resnet50 --n_epochs 100 --gamma 0.1 --dog_group 1 --reweight_groups --asgdro --robust --root_dir /my/dataset/path/MetaDatasetCatDog/ --log_dir /my/log/path --seed 0 

python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --rho 0.5 --model resnet50 --n_epochs 100 --gamma 0.1 --dog_group 2 --reweight_groups --asgdro --robust --root_dir /my/dataset/path/MetaDatasetCatDog/ --log_dir /my/log/path --seed 0 

python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --rho 0.5 --model resnet50 --n_epochs 100 --gamma 0.1 --dog_group 3 --reweight_groups --asgdro --robust --root_dir /my/dataset/path/MetaDatasetCatDog/ --log_dir /my/log/path --seed 0 

python run_expt.py -s confounder -d MetaDatasetCatDog -t cat -c background --lr 0.001 --batch_size 16 --weight_decay 0.0001 --rho 0.5 --model resnet50 --n_epochs 100 --gamma 0.1 --dog_group 4 --reweight_groups --asgdro --robust --root_dir /my/dataset/path/MetaDatasetCatDog/ --log_dir /my/log/path --seed 0 
```




## Domain Shifts
To run the code, you need to first enter the directory: `cd domain_shifts`.
When you run your command, please specify your own $DATA_PATH with "--data-dir".

The datasets will be automatically downloaded when running the scripts provided below. 

#### CivilComments

```
python main.py --dataset civil --algorithm asgdro --data-dir /$DATA_PATH/CivilComments 
```

#### Camelyon17
```
python main_dg.py --dataset camelyon --algorithm asgdro --data-dir /my/dataset/path/Camelyon17 --experiment /my/expeirment/name/ --balancing_sampling --seed 0
```

#### FMoW
```
 python main.py --dataset fmow --algorithm asgdro --data-dir /my/dataset/path/FMoW --experiment /my/experiment/name/ --balancing_sampling --rho 0.05 --epochs 10 --batch_size 64 --seed 0
```

#### RxRx1
```
python main.py --dataset rxrx --algorithm asgdro --data-dir /my/dataset/path/RxRx1 --experiment /my/experiment/name/ --balancing_sampling --rho 0.8 --lr 1e-4 --weight_decay 1e-3 --seed 0 
```

#### Amazon
```
python main.py --dataset amazon --algorithm asgdro --data-dir /my/dataset/path/Amazon --experiment /my/experiment/name/ --balancing_sampling --rho $RHO --lr 2e-6 --weight_decay 0 --seed 0
```


## Domainbed

For Domainbed experiments, refer to `domainbed/README.md`.