from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data 
from torch_geometric.data import Dataset, DataLoader

import numpy as np
import os
import math
import json
import random
import wandb
from datetime import datetime
from tqdm import trange
from sklearn.metrics import precision_score, recall_score

from models.graph_rules import GraphRulesSingleRelationship
from dataloaders.scitsr_graph_rules import ScitsrGraphRules
from misc.args import scitsr_params, base_params, trainer_params, evaluation_params
from ops.utils import cal_adj_label
from ops.misc import weights_init, mkdir_p


def main(config):
    dataset_params = config['dataset_params']
    base_params = config['base_params']
    trainer_params = config['trainer_params']
    eval_params = config['eval_params']

    # Define dataloader
    eval_dataset = ScitsrGraphRules(dataset_params, partition='eval')
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False,
                             num_workers=8, pin_memory=True)

    # Define model and initialize weights
    if dataset_params.gr_single_relationship:
        row_model = GraphRulesSingleRelationship(base_params)
        col_model = GraphRulesSingleRelationship(base_params)
        
        row_model_path = eval_params.pretrained_root + os.sep \
                         + 'row_only_net_{}_best_so_far.pth'.format(eval_params.best_epoch)
        row_model.load_state_dict(torch.load(row_model_path))
        row_model.to(DEVICE)

        col_model_path = eval_params.pretrained_root + os.sep \
                         + 'col_only_net_{}_best_so_far.pth'.format(eval_params.best_epoch)
        col_model.load_state_dict(torch.load(col_model_path))
        col_model.to(DEVICE)

if __name__ == "__main__":
    dataset_params = scitsr_params()
    base_params = base_params()
    eval_params = evaluation_params()
    trainer_params = trainer_params()

    # assign datadir
    dataset_params.data_dir = eval_params.data_dir
    dataset_params.new_imglist = True

    # Seed things
    torch.manual_seed(trainer_params.seed)
    torch.cuda.manual_seed(trainer_params.seed)
    np.random.seed(trainer_params.seed)
    random.seed(trainer_params.seed)

    # Create save locations
    time = datetime.now()
    time = time.strftime('%Y_%m_%d_%H_%M')
    root_path = os.getcwd() + os.sep + eval_params.exp + os.sep + trainer_params.run + '_' + time
    mkdir_p(root_path)
    results_path = root_path + os.sep + 'results'
    mkdir_p(results_path)

    # Set CUDA access
    use_cuda = torch.cuda.is_available() and trainer_params.device == 'cuda'
    if use_cuda:
        DEVICE = torch.device("cuda")
    else:
        print('Warning: CPU is being used to run model,\
             CUDA device not being used for current run')
        DEVICE = torch.device("cpu")

    namespace_config_dict = {
                                'eval_params': eval_params,
                                'base_params': base_params,
                                'trainer_params': trainer_params,
                                'dataset_params': dataset_params
                            }

    print("#" * 100)
    print("CURRENT CONFIGURATION")
    print("#" * 100)
    print(namespace_config_dict)
    print("#" * 100)

    main(config=namespace_config_dict)