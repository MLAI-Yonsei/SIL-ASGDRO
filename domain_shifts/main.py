import argparse
import datetime
import time
import json
import os
import pdb
import sys
import csv
import tqdm
from collections import defaultdict
from transformers import get_cosine_schedule_with_warmup
from tempfile import mkdtemp
import ipdb

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F

import models
from config import dataset_defaults
from utils import unpack_data, sample_domains, save_best_model, set_seed_everything, \
    Logger, return_predict_fn, return_criterion, return_criterion_with_logits, fish_step, args_name

from pytorch_transformers import AdamW, WarmupLinearSchedule
from mixup import mix_forward, bert_mix_forward, rand_bbox, mix_up

from sam import SAM
from bypass_bn import enable_running_stats, disable_running_stats

runId = datetime.datetime.now().isoformat().replace(':', '_')
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Gradient Matching for Domain Generalization.')
# General
parser.add_argument('--dataset', type=str, default='civil',
                    help="Name of dataset, choose from amazon, camelyon, "
                         "civil, fmow, rxrx")
parser.add_argument('--algorithm', type=str, default='erm',
                    help='training scheme, choose between fish or erm.')
parser.add_argument('--experiment', type=str, default='.',
                    help='experiment name, set as . for automatic naming.')
parser.add_argument('--data-dir', type=str, default='./',
                    help='path to data dir')
parser.add_argument('--stratified', action='store_true', default=False,
                    help='whether to use stratified sampling for classes')
parser.add_argument('--num-domains', type=int, default=15,
                    help='Number of domains, only specify for cdsprites')
# Computation
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA use')
parser.add_argument('--seed', type=int, default=-1,
                    help='random seed, set as -1 for random.') #TODO
parser.add_argument("--n_groups_per_batch", default=4, type=int)
parser.add_argument("--cut_mix", default=False, action='store_true')
parser.add_argument("--mix_alpha", default=2, type=float)
parser.add_argument("--print_loss_iters", default=100, type=int)
parser.add_argument("--group_by_label", default=False, action='store_true')
parser.add_argument("--power", default=0, type=float)
parser.add_argument("--reweight_groups", default=False, action='store_true')
parser.add_argument("--balancing_sampling", default=False, action='store_true')
parser.add_argument("--eval_batch_size", default=256, type=int)
parser.add_argument("--scheduler", default=None, type=str)

# parameters about training twice
parser.add_argument("--save_pred", default=False, action='store_true')
parser.add_argument("--save_dir", default='result', type=str)
parser.add_argument("--use_bert_params", default=1, type=int)
parser.add_argument("--max_grad_norm", default=1.0, type=float)
parser.add_argument("--warmup_steps", default=0, type=int)

# ASGDRO
parser.add_argument("--rho", default=0.5, type=float)

# Clipping
parser.add_argument("--clip_first", default=False, action='store_true')
parser.add_argument("--clip_second", default=False, action='store_true')

# TODO Config
parser.add_argument("--epochs", default=None, type=int)
parser.add_argument("--batch_size", default=None, type=int) ##
parser.add_argument("--optimiser", default=None, type=str, choices = ['SGD','Adam','AdamW'])
parser.add_argument("--lr", default=None, type=float) ##
parser.add_argument("--weight_decay", default=None, type=float) ##
parser.add_argument("--adjust", default=0, type=int)
parser.add_argument("--robust_step_size", default=0.01, type=float)

# Misc
parser.add_argument("--debug", default=False, action='store_true')

args = parser.parse_args()

name = args_name(args)

#### Avoid config for sweep
temp_epochs = args.epochs
temp_optimiser = args.optimiser
temp_batch_size = args.batch_size
temp_lr = args.lr
temp_weight_decay = args.weight_decay
temp_warmup_steps = args.warmup_steps
####

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

args_dict = args.__dict__
args_dict.update(dataset_defaults[args.dataset]) ##
args = argparse.Namespace(**args_dict)

## Sweep
if temp_optimiser != None:
    args.optimiser = temp_optimiser
if temp_epochs != None:
    args.epochs = temp_epochs
if temp_batch_size != None:
    args.batch_size = temp_batch_size
if temp_lr != None:
    args.optimiser_args["lr"] = temp_lr
if temp_weight_decay != None:
    args.optimiser_args["weight_decay"] = temp_weight_decay
if (temp_warmup_steps != None) and (args.dataset == "rxrx"):
    args.scheduler_kwargs["num_warmup_steps"] = temp_warmup_steps

##
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)


# Choosing and saving a random seed for reproducibility
if args.seed == -1:
    args.seed = int(torch.randint(0, 2 ** 32 - 1, (1,)).item())

# Set seed everything for reproducibility
set_seed_everything(args.seed)
print("Current Seed: ", args.seed) # TODO Make pretty

# experiment directory setup
args.experiment = f"{args.dataset}_{args.algorithm}_{args.seed}" \
    if args.experiment == '.' else args.experiment
args.name = args_name(args)

if args.dataset == 'amazon' and args.model == 'distilbert':
    args.experiment += '_distillbert'

directory_name_ = '../experiments/{}'.format(os.path.join(args.experiment))
csv_path = os.path.join(directory_name_, "results.csv")
directory_name = os.path.join(directory_name_, args.name)

if not os.path.exists(directory_name):
    os.makedirs(directory_name)
else:
    print("Already Runned {}!!".format(directory_name))
    sys.exit(0)

# runPath = mkdtemp(prefix=runId, dir=directory_name)
runPath = directory_name

# logging setup
sys.stdout = Logger('{}/run.log'.format(runPath))
print('RunID:' + runPath)
print(args)
with open('{}/args.json'.format(runPath), 'w') as fp:
    json.dump(args.__dict__, fp)
torch.save(args, '{}/args.rar'.format(runPath))

# load model
modelC = getattr(models, args.dataset)
if args.algorithm == 'lisa' or args.algorithm == 'lisa_asgdro': args.batch_size //= 2
if args.dataset == 'camelyon': args.n_groups_per_batch = 3
train_loader, tv_loaders = modelC.getDataLoaders(args, device=device)
val_loader, test_loader = tv_loaders['val'], tv_loaders['test']
model = modelC(args, weights=None).to(device)

n_class = getattr(models, f"{args.dataset}_n_class")
n_domain = getattr(models, f"{args.dataset}_n_domain") #TODO: Set all other dataset in domain shift

# TODO: Current Complete = [Civil, ]
# if args.group_by_label:
#     n_domain = 2

assert args.optimiser in ['SGD', 'Adam', 'AdamW'], "Invalid choice of optimiser, choose between 'Adam' and 'SGD'"
# TODO: Check Civil or bert-based model, Adam(LISA: AMSgrad) or AdamW(Wilds)
opt = getattr(optim, args.optimiser)
#opt = optim.AdamW # TODO origianl - adam #########################################################
if args.use_bert_params and args.dataset == 'civil': #TODO: Need Change During Parameter Sweep
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay":
                args.weight_decay, #TODO
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay":
                0.0,
        },
    ]
    if args.algorithm == 'asgdro' or args.algorithm == "lisa_asgdro":
        sam_optimizer = SAM(optimizer_grouped_parameters, opt, **args.optimiser_args,
                adaptive = True, rho = args.rho) # TODO
    else:
        optimiserC = opt(optimizer_grouped_parameters, **args.optimiser_args)
else:
    if args.algorithm == 'asgdro' or args.algorithm == "lisa_asgdro":
        sam_optimizer = SAM(model.parameters(), opt, **args.optimiser_args,
                adaptive =True, rho = args.rho) # TODO
    else:
        optimiserC = opt(model.parameters(), **args.optimiser_args)

predict_fn = return_predict_fn(args.dataset)

if args.algorithm.endswith('dro'):
    criterion = return_criterion_with_logits(args.dataset)
else:
    criterion = return_criterion(args.dataset)

def split_into_groups(g):
    """
    Args:
        - g (Tensor): Vector of groups
    Returns:
        - groups (Tensor): Unique groups present in g
        - group_indices (list): List of Tensors, where the i-th tensor is the indices of the
                                elements of g that equal groups[i].
                                Has the same length as len(groups).
        - unique_counts (Tensor): Counts of each element in groups.
                                 Has the same length as len(groups).
    """
    unique_groups, unique_counts = torch.unique(g, sorted=False, return_counts=True)
    group_indices = []
    for group in unique_groups:
        group_indices.append(
            torch.nonzero(g == group, as_tuple=True)[0])
    return unique_groups, group_indices, unique_counts


def train_erm(train_loader, epoch, agg):
    running_loss = 0
    total_iters = len(train_loader)
    print('\n====> Epoch: {:03d} '.format(epoch))
    for i, data in enumerate(train_loader):
        model.train()
        # get the inputs
        x, y = unpack_data(data, device)
        optimiserC.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        if args.use_bert_params:
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           args.max_grad_norm)
        optimiserC.step()
        if scheduler is not None:
            scheduler.step()
        running_loss += loss.item()
        # print statistics
        if (i + 1) % args.print_loss_iters == 0:
            agg['train_loss'].append(running_loss / args.print_loss_iters)
            agg['train_iters'].append(i + 1 + epoch * total_iters)
            print(
                'iteration {:05d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters, running_loss / args.print_loss_iters))
            running_loss = 0.0

        if (i + 1) % args.print_iters == 0 and args.print_iters != -1 and (
                (i + 1) / args.print_iters >= 2 or epoch >= 1):
            test(val_loader, agg, loader_type='val')
            test(test_loader, agg, loader_type='test')
            save_best_model(model, runPath, agg)


def train_groupdro(train_loader, epoch, agg):
    model.train()
    running_loss = 0
    total_iters = len(train_loader)
    adjustment = args.adjust/torch.sqrt(args.group_count).to(device)
    if epoch == 0:
        agg['adv_probs'] = torch.ones(agg['n_domain']).to(device)/agg['n_domain']
    adv_probs = agg['adv_probs'].to(device)
    print('\n====> Epoch: {:03d} '.format(epoch))
    for i, data in enumerate(train_loader):
        model.train()
        # get the inputs with group index
        x, y, g, _ = data[0].to(device), data[1].to(device), data[2].to(device), data[3]
        g = torch.tensor([train_loader.dataset.domain2idx[g[k].item()] for k in range(len(g))]).cuda()
        group_list = torch.arange(n_domain).to(device)
        group_map = (group_list.view(-1,1)==g)
        num_each_group = group_map.float().sum(dim=1)
        y_hat = model(x)
        logits = criterion(y_hat, y)
        group_loss = (group_map*logits).sum(dim=1)
        group_loss = group_loss/(num_each_group + (num_each_group==0).float()) # Avoid nans
        group_loss += adjustment

        # Update Weight of Each Group Loss
        adv_probs = adv_probs * torch.exp(args.robust_step_size*group_loss.data)
        adv_probs = adv_probs/adv_probs.sum()

        #Robust Loss
        robust_loss = torch.dot(adv_probs,group_loss)

        #Update
        optimiserC.zero_grad()
        robust_loss.backward()
        if args.use_bert_params:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimiserC.step()

        #Renew
        if scheduler is not None:
            scheduler.step()
        running_loss += robust_loss.item()

        if (i + 1) % args.print_loss_iters == 0:
            agg['train_loss'].append(running_loss / args.print_loss_iters)
            agg['train_iters'].append(i + 1 + epoch * total_iters)
            print('iteration {:05d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters,
                                                                running_loss / args.print_loss_iters))
            running_loss = 0.0

        if (i + 1) % args.print_iters == 0 and args.print_iters != -1 and (
                (i + 1) / args.print_iters >= 2 or epoch >= 1):
            test(val_loader, agg, loader_type='val')
            test(test_loader, agg, loader_type='test')
            save_best_model(model, runPath, agg)
    agg['adv_probs'] = adv_probs.detach().cpu()

def train_asgdro(train_loader, epoch, agg):
    model.train()
    running_loss = 0
    total_iters = len(train_loader)
    adjustment = args.adjust/torch.sqrt(args.group_count).to(device)
    if epoch == 0:
        agg['adv_probs'] = torch.ones(agg['n_domain'])/agg['n_domain']
    adv_probs = agg['adv_probs'].to(device)
    print('\n====> Epoch: {:03d} '.format(epoch))
    for i, data in enumerate(train_loader):
        model.train()
        # get the inputs with group index
        x, y, g, _ = data[0].to(device), data[1].to(device), data[2].to(device), data[3]
        g = torch.tensor([train_loader.dataset.domain2idx[g[k].item()] for k in range(len(g))]).cuda()
        # First Step (Find Perturbed Weights for Empirical Risk Minimization [ERM])
        enable_running_stats(model)
        y_hat = model(x)
        loss = criterion(y_hat, y).mean()
        loss.backward()
        if args.clip_first:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        sam_optimizer.first_step(zero_grad=True)
        
        # Prepare Second Step
        disable_running_stats(model)
        group_list = torch.arange(n_domain).to(device)
        group_map = (group_list.view(-1,1)==g)
        num_each_group = group_map.float().sum(dim=1)
        y_hat = model(x)
        logits = criterion(y_hat, y)
        group_loss = (group_map*logits).sum(dim=1)
        group_loss = group_loss/(num_each_group + (num_each_group==0).float()) # Avoid nans
        group_loss += adjustment

        adv_probs = adv_probs * torch.exp(args.robust_step_size*group_loss.data)
        adv_probs = adv_probs/adv_probs.sum()

        #Robust Loss
        robust_loss = torch.dot(adv_probs,group_loss)

        #Update
        robust_loss.backward()

        if args.clip_second:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        sam_optimizer.second_step(zero_grad=True)

        #Log
        if scheduler is not None:
            scheduler.step()
        running_loss += robust_loss.item()

        if (i + 1) % args.print_loss_iters == 0:
            agg['train_loss'].append(running_loss / args.print_loss_iters)
            agg['train_iters'].append(i + 1 + epoch * total_iters)
            print('iteration {:05d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters,
                                                                running_loss / args.print_loss_iters))
            running_loss = 0.0

        if (i + 1) % args.print_iters == 0 and args.print_iters != -1 and (
                (i + 1) / args.print_iters >= 2 or epoch >= 1):
            test(val_loader, agg, loader_type='val')
            test(test_loader, agg, loader_type='test')
            save_best_model(model, runPath, agg)
    agg['adv_probs'] = adv_probs.detach().cpu()

def train_lisa(train_loader, epoch, agg):
    model.train()
    train_loader.dataset.reset_batch()
    running_loss = 0
    total_iters = len(train_loader)
    print('\n====> Epoch: {:03d} '.format(epoch))

    # The probabilities for each group do not equal to each other.
    for i, data in enumerate(train_loader):
        model.train()
        x1, y1, g1, prev_idx = data[0].to(device), data[1].to(device), data[2].to(device), data[3]
        x2, y2, g2 = [], [], []
        for g, idx in zip(g1, prev_idx):
            tmp_x, tmp_y, tmp_g = train_loader.dataset.get_sample(g, idx.item())
            x2.append(tmp_x.unsqueeze(0))
            y2.append(tmp_y)
            g2.append(tmp_g)
        
        x2 = torch.cat(x2).to(device)
        y2 = torch.stack(y2).reshape(-1).to(device)
        y1_onehot = torch.zeros(len(y1), n_class).to(y1.device)
        y1 = y1_onehot.scatter_(1, y1.unsqueeze(1), 1)

        y2_onehot = torch.zeros(len(y2), n_class).to(y2.device)
        y2 = y2_onehot.scatter_(1, y2.unsqueeze(1), 1)

        if args.dataset in ['civil', 'amazon']:
            x = torch.cat([x1, x2]).to(device)
            y_onehot = torch.cat([y1_onehot, y2_onehot])
            x = model.model[0](x)
            bsz = len(x)
            x, mixed_y = mix_up(args, x, y_onehot, torch.cat([x[bsz // 2:], x[:bsz // 2]]),
                                torch.cat([y_onehot[bsz // 2:], y_onehot[:bsz // 2]]))
            outputs = model.model[1](x)

        else:

            if args.cut_mix:
                rand_index = torch.cat([torch.arange(len(y2)) + len(y1), torch.arange(len(y1))])
                lam = np.random.beta(args.mix_alpha, args.mix_alpha)

                input = torch.cat([x1, x2])
                target = torch.cat([y1, y2])

                target_a = target
                target_b = target[rand_index]

                bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
                input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
                mixed_y = lam * target_a + (1 - lam) * target_b
                outputs = model(input)
            else:
                mixed_x1, mixed_y1 = mix_up(args, x1, y1, x2, y2)
                mixed_x2, mixed_y2 = mix_up(args, x2, y2, x1, y1)
                mixed_x = torch.cat([mixed_x1, mixed_x2])
                mixed_y = torch.cat([mixed_y1, mixed_y2])
                outputs = model(mixed_x)

        loss = - F.log_softmax(outputs, dim=-1) * mixed_y
        loss = loss.sum(-1)
        loss = loss.mean()
        optimiserC.zero_grad()
        loss.backward()
        optimiserC.step()
        if scheduler is not None:
            scheduler.step()
        running_loss += loss.item()

        if (i + 1) % args.print_loss_iters == 0:
            agg['train_loss'].append(running_loss / args.print_loss_iters)
            agg['train_iters'].append(i + 1 + epoch * total_iters)
            print('iteration {:05d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters,
                                                                running_loss / args.print_loss_iters))
            running_loss = 0.0

        if (i + 1) % args.print_iters == 0 and args.print_iters != -1 and (
                (i + 1) / args.print_iters >= 2 or epoch >= 1):
            test(val_loader, agg, loader_type='val')
            test(test_loader, agg, loader_type='test')
            save_best_model(model, runPath, agg)

def train_lisa_asgdro(train_loader, epoch, agg):
    model.train()
    train_loader.dataset.reset_batch()
    running_loss = 0
    adjustment = args.adjust/torch.sqrt(args.group_count).to(device)
    total_iters = len(train_loader)
    if epoch == 0:
        agg['adv_probs'] = torch.ones(agg['n_domain'])/agg['n_domain']
    adv_probs = agg['adv_probs'].to(device)
    print('\n====> Epoch: {:03d} '.format(epoch))
    
    # The probabilities for each group do not equal to each other.
    for i, data in enumerate(train_loader):
        model.train()
        x1, y1, g1, prev_idx = data[0].to(device), data[1].to(device), data[2].to(device), data[3]
        x2, y2, g2 = [], [], []
        for gg, idx in zip(g1, prev_idx):
            tmp_x, tmp_y, tmp_g = train_loader.dataset.get_sample(gg, idx.item())
            x2.append(tmp_x.unsqueeze(0))
            y2.append(tmp_y)
            g2.append(tmp_g)
        
        x2 = torch.cat(x2).to(device)
        y2 = torch.stack(y2).reshape(-1).to(device)
        y1_onehot = torch.zeros(len(y1), n_class).to(y1.device)
        y1 = y1_onehot.scatter_(1, y1.unsqueeze(1), 1)

        y2_onehot = torch.zeros(len(y2), n_class).to(y2.device)
        y2 = y2_onehot.scatter_(1, y2.unsqueeze(1), 1)
        enable_running_stats(model)
        if args.dataset in ['civil', 'amazon']: 
            x = torch.cat([x1, x2]).to(device)
            y_onehot = torch.cat([y1_onehot, y2_onehot])
            x = model.model[0](x)
            bsz = len(x)
            x, mixed_y = mix_up(args, x, y_onehot, torch.cat([x[bsz // 2:], x[:bsz // 2]]),
                                torch.cat([y_onehot[bsz // 2:], y_onehot[:bsz // 2]]))
            outputs = model.model[1](x)
            g = torch.concat([g1,g1])
        else:

            if args.cut_mix:
                rand_index = torch.cat([torch.arange(len(y2)) + len(y1), torch.arange(len(y1))])
                lam = np.random.beta(args.mix_alpha, args.mix_alpha)

                input = torch.cat([x1, x2])
                target = torch.cat([y1, y2])

                target_a = target
                target_b = target[rand_index]

                bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
                input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
                mixed_y = lam * target_a + (1 - lam) * target_b
                outputs = model(input)
            else:
                mixed_x1, mixed_y1 = mix_up(args, x1, y1, x2, y2)
                mixed_x2, mixed_y2 = mix_up(args, x2, y2, x1, y1)
                mixed_x = torch.cat([mixed_x1, mixed_x2])
                mixed_y = torch.cat([mixed_y1, mixed_y2])
                outputs = model(mixed_x)

        loss = - F.log_softmax(outputs, dim=-1) * mixed_y
        
        loss = loss.sum(-1)
        loss = loss.mean()
        loss.backward()
        if args.clip_first:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        sam_optimizer.first_step(zero_grad=True)

        disable_running_stats(model)
        group_list = torch.arange(n_domain).to(device)
        group_map = (group_list.view(-1,1)==g)
        num_each_group = group_map.float().sum(dim=1)

        if args.dataset in ['civil', 'amazon']: 
            x = torch.cat([x1, x2]).to(device)
            y_onehot = torch.cat([y1_onehot, y2_onehot])
            x = model.model[0](x)
            bsz = len(x)
            x, mixed_y = mix_up(args, x, y_onehot, torch.cat([x[bsz // 2:], x[:bsz // 2]]),
                                torch.cat([y_onehot[bsz // 2:], y_onehot[:bsz // 2]]))
            outputs = model.model[1](x)
            g = torch.concat([g1,g1])
        else:
            if args.cut_mix:
                rand_index = torch.cat([torch.arange(len(y2)) + len(y1), torch.arange(len(y1))])
                lam = np.random.beta(args.mix_alpha, args.mix_alpha)

                input = torch.cat([x1, x2])
                target = torch.cat([y1, y2])

                target_a = target
                target_b = target[rand_index]

                bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
                input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
                mixed_y = lam * target_a + (1 - lam) * target_b
                outputs = model(input)
            else:
                mixed_x1, mixed_y1 = mix_up(args, x1, y1, x2, y2)
                mixed_x2, mixed_y2 = mix_up(args, x2, y2, x1, y1)
                mixed_x = torch.cat([mixed_x1, mixed_x2])
                mixed_y = torch.cat([mixed_y1, mixed_y2])
                outputs = model(mixed_x)

        loss = - F.log_softmax(outputs, dim=-1) * mixed_y
        loss = loss.sum(-1)
        group_loss = (group_map*loss).sum(dim=1)
        group_loss = group_loss/(num_each_group + (num_each_group==0).float()) # Avoid nans
        group_loss += adjustment
        adv_probs = adv_probs * torch.exp(args.robust_step_size*group_loss.data)
        adv_probs = adv_probs/adv_probs.sum()

        #Robust Loss
        loss = torch.dot(adv_probs,group_loss)

        #Update
        loss.backward()

        if args.clip_second:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        sam_optimizer.second_step(zero_grad=True)

        if scheduler is not None:
            scheduler.step()
        running_loss += loss.item()

        if (i + 1) % args.print_loss_iters == 0:
            agg['train_loss'].append(running_loss / args.print_loss_iters)
            agg['train_iters'].append(i + 1 + epoch * total_iters)
            print('iteration {:05d}/{:d}: loss: {:6.3f}'.format(i + 1, total_iters,
                                                                running_loss / args.print_loss_iters))
            running_loss = 0.0

        if (i + 1) % args.print_iters == 0 and args.print_iters != -1 and (
                (i + 1) / args.print_iters >= 2 or epoch >= 1):
            test(val_loader, agg, loader_type='val')
            test(test_loader, agg, loader_type='test')
            save_best_model(model, runPath, agg)

    agg['adv_probs'] = adv_probs.detach().cpu()


def save_pred(model, train_loader, epoch, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    yhats, ys, idxes = [], [], []
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            model.eval()
            x, y, idx = data[0].to(device), data[1].to(device), data[-1].to(device)
            y_hat = model(x)
            ys.append(y.cpu())
            yhats.append(y_hat.cpu())
            idxes.append(idx.cpu())

        ypreds, ys, idxes = predict_fn(torch.cat(yhats)), torch.cat(ys), torch.cat(idxes)

        ypreds = ypreds[torch.argsort(idxes)]
        ys = ys[torch.argsort(idxes)]

        y = torch.cat([ys.reshape(-1, 1), ypreds.reshape(-1, 1)], dim=1)

        df = pd.DataFrame(y.cpu().numpy(), columns=['y_true', 'y_pred'])
        df.to_csv(os.path.join(save_dir, f"{args.dataset}_{args.algorithm}_{epoch}.csv"))

        # print accuracy
        wrong_labels = (df['y_true'].values == df['y_pred'].values).astype(int)
        from wilds.common.grouper import CombinatorialGrouper
        grouper = CombinatorialGrouper(train_loader.dataset.dataset.dataset, ['y', 'black'])
        group_array = grouper.metadata_to_group(train_loader.dataset.dataset.dataset.metadata_array).numpy()
        group_array = group_array[np.where(
            train_loader.dataset.dataset.dataset.split_array == train_loader.dataset.dataset.dataset.split_dict[
                'train'])]
        for i in np.unique(group_array):
            idxes = np.where(group_array == i)[0]
            print(f"domain = {i}, length = {len(idxes)}, acc = {np.sum(wrong_labels[idxes] / len(idxes))} ")

        def print_group_info(idxes):
            group_ids, group_counts = np.unique(group_array[idxes], return_counts=True)
            for idx, j in enumerate(group_ids):
                print(f"group[{j}]: {group_counts[idx]} ")

        correct_idxes = np.where(wrong_labels == 1)[0]
        print("correct points:")
        print_group_info(correct_idxes)
        wrong_idxes = np.where(wrong_labels == 0)[0]
        print("wrong points:")
        print_group_info(wrong_idxes)


def test(test_loader, agg, loader_type='test', verbose=True, save_ypred=False, save_dir=None):
    model.eval()
    yhats, ys, metas = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            # get the inputs
            x, y = batch[0].to(device), batch[1].to(device)
            y_hat = model(x)
            ys.append(y)
            yhats.append(y_hat)
            metas.append(batch[2])

        ypreds, ys, metas = predict_fn(torch.cat(yhats)), torch.cat(ys), torch.cat(metas)
        if save_ypred:
            if args.dataset == 'poverty':
                save_name = f"{args.dataset}_split:{loader_type}_fold:" \
                            f"{['A', 'B', 'C', 'D', 'E'][args.seed]}" \
                            f"_epoch:best_pred.csv"
            else:
                save_name = f"{args.dataset}_split:{loader_type}_seed:" \
                            f"{args.seed}_epoch:best_pred.csv"
            with open(f"{runPath}/{save_name}", 'w') as f:
                writer = csv.writer(f)
                writer.writerows(ypreds.unsqueeze(1).cpu().tolist())
        test_val = test_loader.dataset.eval(ypreds.cpu(), ys.cpu(), metas)
        if args.dataset == 'poverty':
            with open(f"{runPath}/{save_name}_ys", 'w') as f:
                writer = csv.writer(f)
                writer.writerows(ys.unsqueeze(1).cpu().tolist())
        agg[f'{loader_type}_stat'].append(test_val[0][args.selection_metric])
        if verbose:
            print(f"=============== {loader_type} ===============\n{test_val[-1]}")
        if save_ypred:
            return test_val[0]



if __name__ == '__main__':

    # Start Time
    start_time = time.time()

    if args.scheduler == 'cosine_schedule_with_warmup':
        if args.algorithm == "asgdro" or args.algorithm == 'lisa_asgdro':
            scheduler = get_cosine_schedule_with_warmup(
                    sam_optimizer.base_optimizer,
                    num_training_steps=len(train_loader) * args.epochs,
                    **args.scheduler_kwargs)

        else:
            scheduler = get_cosine_schedule_with_warmup(
                optimiserC,
                num_training_steps=len(train_loader) * args.epochs,
                **args.scheduler_kwargs)
        print("scheduler has been defined")
    elif args.scheduler == 'linear_scheduler':
        t_total = len(train_loader) * args.epochs
        print(f"\nt_total is {t_total}\n")
        if args.algorithm == "asgdro" or args.algorithm == 'lisa_asgdro':
            scheduler = WarmupLinearSchedule(sam_optimizer.base_optimizer,
                    warmup_steps=args.warmup_steps, t_toal=t_total)
        else:
            scheduler = WarmupLinearSchedule(optimiserC,
                                         warmup_steps=args.warmup_steps,
                                         t_total=t_total)
    else:
        scheduler = None

    print(
        "=" * 30 + f"Training: {args.algorithm}" + "=" * 30)

    # Start Time
    start_time = time.time()

    train = locals()[f'train_{args.algorithm}']
    agg = defaultdict(list)
    agg['val_stat'] = [0.]
    agg['test_stat'] = [0.]
    agg['n_domain'] = n_domain
    agg['name'] = args.name
    for epoch in range(args.epochs):
        train(train_loader, epoch, agg)
        test(val_loader, agg, loader_type='val')
        test(test_loader, agg, loader_type='test')
        save_best_model(model, runPath, agg)
        if args.save_pred:
            save_pred(model, train_loader, epoch, args.save_dir)
    
    # End Time
    end_time = time.time()
    exec_time = end_time - start_time

    if args.debug:
        modelC = getattr(models, args.dataset)
    else: 
        model.load_state_dict(torch.load(runPath + '/model.rar'))
        
    print('Finished training! Loading best model...')
    for split, loader in tv_loaders.items():
        res_dict = test(loader, agg, loader_type=split, save_ypred=True)
    df = pd.DataFrame([res_dict])
    if not os.path.exists(csv_path):
        df.to_csv(csv_path)
    else: 
        final_df = pd.read_csv(csv_path, index_col=0)
        final_df = pd.concat([final_df, df])
        final_df.to_csv(csv_path)

    print("Time for Execution: ", exec_time)
