import os
from copy import deepcopy

import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification
from transformers import BertTokenizerFast, AutoTokenizer
from transformers import logging
from .bert.bert import BertFeaturizer
from .bert.distilbert import DistilBertFeaturizer
from .bert.ibert import IBertFeaturizer
from .bert.toxbert import ToxFeaturizer
from wilds.common.data_loaders import get_eval_loader
from torch.utils.data.sampler import WeightedRandomSampler
from wilds.datasets.civilcomments_dataset import CivilCommentsDataset
from transformers import DistilBertTokenizerFast
from .datasets import CivilComments_Batched_Dataset, seed_worker

# logging.set_verbosity_error()

MAX_TOKEN_LENGTH = 300
NUM_CLASSES = 2
NUM_DOMAINS = 4  # # of {(toxic, black)}

def initialize_bert_transform(args):
    """Adapted from the Wilds library, available at: https://github.com/p-lambda/wilds"""
    if args.model == 'bert':
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    elif args.model == 'distilbert':
        tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    elif args.model == 'ibert':
        tokenizer = AutoTokenizer.from_pretrained("kssteven/ibert-roberta-base")
    elif args.model == 'toxbert':
        tokenizer = AutoTokenizer.from_pretrained("rungalileo/toxic-bert-quantized-traced")

    
    def transform(text):
        tokens = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            return_tensors='pt')
        if args.model == 'bert':
            x = torch.stack(
                (tokens['input_ids'],
                 tokens['attention_mask'],
                 tokens['token_type_ids']),
                dim=2)
        elif args.model == 'distilbert':
            x = torch.stack(
                (tokens['input_ids'],
                 tokens['attention_mask']),
                dim=2)
        elif args.model == 'ibert':
            x = torch.stack(
                (tokens['input_ids'],
                 tokens['attention_mask']),
                 # IBERT tokenizer doesn't include token_type_ids, so we only stack input_ids and attention_mask.
                dim=2
            )
        elif args.model == 'toxbert':
            x = torch.stack(
                (tokens['input_ids'],
                 tokens['attention_mask']),
                dim=2
            )
        x = torch.squeeze(x, dim=0) # First shape dim is always 1
        return x
    return transform

class BertClassifier(BertForSequenceClassification):
    """Adapted from the Wilds library, available at: https://github.com/p-lambda/wilds"""
    def __init__(self, args):
        super().__init__(args)
        self.d_out = 2

    def __call__(self, x):
        input_ids = x[:, :, 0]
        attention_mask = x[:, :, 1]
        token_type_ids = x[:, :, 2]
        outputs = super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )[0]
        return outputs

class Model(nn.Module):
    def __init__(self, args, weights):
        super(Model, self).__init__()
        self.num_classes = NUM_CLASSES
        # self.model = BertClassifier.from_pretrained(
        #     'bert-base-uncased',
        #     num_labels=2,
        # )
        if args.model == 'bert':
            featurizer = BertFeaturizer.from_pretrained("bert-base-uncased")
        elif args.model == 'distilbert':
            featurizer = DistilBertFeaturizer.from_pretrained("distilbert-base-uncased")
        elif args.model == 'ibert':
            featurizer = IBertFeaturizer.from_pretrained("kssteven/ibert-roberta-base")
            # featurizer = ToxFeaturizer.from_pretrained("rungalileo/toxic-bert-quantized-traced")
        else:
            raise NotImplementedError
        classifier = nn.Linear(featurizer.d_out, 2)
        self.model = nn.Sequential(featurizer, classifier)

        if weights is not None:
            self.load_state_dict(deepcopy(weights))

    def reset_weights(self, weights):
        self.load_state_dict(deepcopy(weights))

    @staticmethod
    def getDataLoaders(args, device):
        dataset = CivilCommentsDataset(root_dir=args.data_dir, download=True)
        # get all train data
        transform = initialize_bert_transform(args)
        train_data = dataset.get_subset('train', transform=transform)
        # separate into subsets by distribution
        train_sets = CivilComments_Batched_Dataset(args, train_data, batch_size=args.batch_size)
        # take subset of test and validation, making sure that only labels appeared in train
        # are included
        datasets = {}
        for split in dataset.split_dict:
            if split != 'train':
                datasets[split] = dataset.get_subset(split, transform=transform)
        # get the loaders
        kwargs = {'num_workers': 4, 'pin_memory': True, 'drop_last': False, 'worker_init_fn': seed_worker} \
            if device.type == "cuda" else {}
        train_group_count = train_sets.domain_counts
        # if args.algorithm == "lisa_asgdro":
        #     train_group_count = [train_group_count[0], train_group_count[1]]
        #     args.group_count = torch.Tensor(train_group_count)
        #else:
        args.group_count = torch.Tensor(train_group_count)
        if args.balancing_sampling:
            # For GroupDRO and ASGDRO
            print("Sample instances of each group before balancing")
            # assert train_sets.num_envs == 4
            group_weights = 1/torch.tensor(train_group_count)
            # if args.algorithm == "lisa_asgdro":
            #     weights = group_weights[(train_sets.domains > 0.5).long()]
            # else:
            #     weights = group_weights[train_sets.domains]
            weights = group_weights[train_sets.domains]
            for i in range(len(train_group_count)):
                print(f"Number of Train Instances Group {i} before balancing: {train_group_count[i]}")
            print(f"Number of Total Instances: {sum(train_group_count)}")
            sampler = WeightedRandomSampler(weights, len(train_sets), replacement=True)
            train_loaders = DataLoader(train_sets, batch_size=args.batch_size, sampler=sampler, **kwargs)

        elif args.reweight_groups:
            print(f"upweighting wrong groups by factor {args.upweight_factor}")
            assert args.group_by_wrong
            assert train_sets.num_envs == 2
            group_weights = np.array([args.upweight_factor, 1])
            print(f"Wrong: weight = {group_weights[0]}, numbers = {len(np.where(train_sets.domains == 0)[0])}")
            print(f"Correct: weight = {group_weights[1]}, numbers = {len(np.where(train_sets.domains == 1)[0])}")
            weights = group_weights[train_sets.domains]

            assert len(weights) == len(train_sets)
            sampler = WeightedRandomSampler(weights, len(train_sets), replacement=True)
            train_loaders = DataLoader(train_sets, batch_size=args.batch_size, sampler=sampler, **kwargs)
        else:
            train_loaders = DataLoader(train_sets, batch_size=args.batch_size, shuffle=True, **kwargs)


        tv_loaders = {}
        for split, dataset in datasets.items():
            tv_loaders[split] = get_eval_loader('standard', dataset, batch_size=args.eval_batch_size, **kwargs)
        return train_loaders, tv_loaders

    def forward(self, x):
        return self.model(x)

