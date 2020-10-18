from __future__ import print_function

import torch
import torch.optim as optim
import torch.utils.data 
from torch_geometric.datasets import Dataset, DataLoader
from torch_scatter import scatter_mean
import torch_geometric.transforms as GT

import numpy as np
import os
import math
import json
import wandb

from models.version1 import TbNetV1
from dataloaders.scatter import ScitsrDataset
from misc.args import train_params, scitsr_params, img_model_params, base_params
from ops.misc import weights_init, mkdir_p


def main(config):
    # Define dataloader

    # Define model

    # Define optimizer and learning rate scheduler

    # Watch model

    # Outer train loop
    pass

def train():
    # Train loop for a batch
    pass

def eval():
    # Eval loop 
    pass



if __name__ == "__main__":
    # Get argument dictionaries
    img_params = img_model_params()
    dataset_params = scitsr_params()
    trainer_params = train_params()
    model_base_params = base_params()

    # Seed things
    torch.manual_seed(trainer_params.seed)
    torch.cuda.manual_seed(trainer_params.seed)
    np.random.seed(trainer_params.seed)
    random.seed(trainer_params.seed)

    # Create save locations
    root_path = os.getcwd() + os.sep + os.exp + os.sep + os.run
    mkdir_p(root_path)
    log_path = root_path + os.sep + '/checkpoints'
    mkdir_p(log_path)

    # Set CUDA access
    use_cuda = torch.cuda.is_available() and trainer_params.device == 'cuda'
    if use_cuda:
        trainer_params.device = 'cuda'
    else:
        print('Warning: CPU is being used to run model,\
             CUDA device not being used for current run')
        trainer_params.device = 'cpu'


    # Create wandb config dict
    config_dict = {'img_params': vars(img_params),
                    'dataset_params': vars(dataset_params),
                    'trainer_params': vars(trainer_params),
                    'model_base_params': vars(model_base_params)}
    print("#" * 100)
    print("CURRENT CONFIGURATION")
    print("#" * 100)
    print(config_dict)
    print("#" * 100)

    # Initialize wandb config
    wandb.init(entity='rsaha', project='table_structure_recognition', config=config_dict)

    main(config=config_dict)










