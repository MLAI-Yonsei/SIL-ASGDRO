# DomainBed

## Preparation

### Dependencies

```sh
pip install -r requirements.txt
```

### Datasets

```sh
python -m domainbed.scripts.download --data_dir=/my/datasets/path
```

## How to Run

`train_all.py` script conducts multiple leave-one-out cross-validations for all target domain.

```sh
python train_all.py my_exp_name --dataset VLCS --data_dir /my/datasets/path --trial_seed 0 --seed 0 --algorithm ASGDRO --checkpoint_freq 100 --rho 0.8 --groupdro_eta 1e-2 --lr 1e-5 --weight_decay 1e-6 --resnet_dropout 0.5 --swad False --result_path my_result_name --deterministic
```

Experiment results are reported as a table. In the table, the row `iid` indicates out-of-domain accuracy from SAGM.

Example results (Just Example):
```
+------------+--------------+---------+---------+---------+---------+
| Selection  | art_painting | cartoon |  photo  |  sketch |   Avg.  |
+------------+--------------+---------+---------+---------+---------+
|   oracle   |   87.919%    | 83.209% | 98.278% | 85.305% | 88.678% |
|    iid     |   88.896%    | 80.704% | 97.605% | 80.471% | 86.919% |
|    last    |   84.991%    | 81.397% | 96.482% | 73.187% | 84.014% |
| last (inD) |   96.902%    | 97.476% | 97.213% | 96.275% | 96.967% |
| iid (inD)  |   98.139%    | 97.521% | 97.586% | 97.911% | 97.789% |
+------------+--------------+---------+---------+---------+---------+

```
In this example, the DG performance of ASGDRO for VLCS dataset is 86.919%.

## Citation

Our work is inspired by the following works:

```
@ARTICLE{2020arXiv201001412F,
       author = {{Foret}, Pierre and {Kleiner}, Ariel and {Mobahi}, Hossein and {Neyshabur}, Behnam},
        title = "{Sharpness-Aware Minimization for Efficiently Improving Generalization}",
         year = 2020,
          eid = {arXiv:2010.01412},
       eprint = {2010.01412},
}
```
```
@inproceedings{
zhuang2022surrogate,
title={Surrogate Gap Minimization Improves Sharpness-Aware Training},
author={Juntang Zhuang and Boqing Gong and Liangzhe Yuan and Yin Cui and Hartwig Adam and Nicha C Dvornek and sekhar tatikonda and James s Duncan and Ting Liu},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=edONMAnhLu-}
}
```
```
@inproceedings{cha2021swad,
  title={SWAD: Domain Generalization by Seeking Flat Minima},
  author={Cha, Junbum and Chun, Sanghyuk and Lee, Kyungjae and Cho, Han-Cheol and Park, Seunghyun and Lee, Yunsung and Park, Sungrae},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2021}
}
```
```
@inproceedings{wang2023sharpness,
  title={Sharpness-Aware Gradient Matching for Domain Generalization},
  author={Wang, Pengfei and Zhang, Zhaoxiang and Lei, Zhen and Zhang, Lei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3769--3778},
  year={2023}
}
```


## License

This project includes some code from [DomainBed](https://github.com/facebookresearch/DomainBed/tree/3fe9d7bb4bc14777a42b3a9be8dd887e709ec414), also MIT licensed.

