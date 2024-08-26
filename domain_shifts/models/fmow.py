import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import densenet121
from wilds.common.data_loaders import get_eval_loader
from wilds.datasets.fmow_dataset import FMoWDataset

from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import FMoW_Batched_Dataset, seed_worker

IMG_HEIGHT = 224
NUM_CLASSES = 62
NUM_DOMAINS = 59 # (year, region) in trainset

class Model(nn.Module):
    def __init__(self, args, weights):
        super(Model, self).__init__()
        self.num_classes = NUM_CLASSES
        self.enc = densenet121(pretrained=True).features
        self.classifier = nn.Linear(1024, self.num_classes)
        if weights is not None:
            self.load_state_dict(deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))

    @staticmethod
    def getDataLoaders(args, device):
        dataset = FMoWDataset(root_dir=args.data_dir, download=True)
        # get all train data
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        train_sets = FMoW_Batched_Dataset(args, dataset, 'train', args.batch_size, transform)
        datasets = {}
        for split in dataset.split_dict:
            if split != 'train':
                datasets[split] = dataset.get_subset(split, transform=transform)

        # get the loaders
        kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': False, 'worker_init_fn': seed_worker} \
            if device.type == "cuda" else {}
        train_group_count = train_sets.domain_counts
        args.group_count = torch.Tensor(train_group_count)
        if args.balancing_sampling:
            # For GroupDRO and ASGDRO
            print("Sample instances of each group before balancing")
            assert train_sets.num_envs == 59
            group_weights = 1/torch.tensor(train_group_count)
            domain_idx = [train_sets.domain2idx[i.item()] for i in train_sets.domains] ## TODO: Caution
            weights = group_weights[domain_idx]
            for i in range(len(train_group_count)):
                print(f"Number of Train Instances Domain {i} before balancing: {train_group_count[i]}")
            print(f"Number of Total Instances: {sum(train_group_count)}")
            sampler = WeightedRandomSampler(weights, len(train_sets), replacement=True)
            train_loaders = DataLoader(train_sets, batch_size=args.batch_size, sampler=sampler, **kwargs)

        else:
            train_loaders = DataLoader(train_sets, batch_size=args.batch_size, shuffle=True, **kwargs)
        tv_loaders = {}
        for split, dataset in datasets.items():
            tv_loaders[split] = get_eval_loader('standard', dataset, batch_size=args.eval_batch_size, **kwargs)
        return train_loaders, tv_loaders

    def forward(self, x):
        features = self.enc(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
