from __future__ import print_function

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
from tqdm import tqdm

from models.graph_rules import GraphRulesSingleRelationship, GraphRulesMultiLabel, GraphRulesMultiTask
from dataloaders.scitsr_graph_rules import ScitsrGraphRules
from misc.args import scitsr_params, base_params
from ops.misc import weights_init, mkdir_p

def main(config):
    dataset_params = config['dataset_params']
    base_params = config['base_params']

    # Define dataloader
    train_dataset = ScitsrGraphRules(dataset_params, partition='train')
    train_loader = DataLoader(train_dataset, batch_size=trainer_params.batch_size, shuffle=True)

    val_dataset = ScitsrGraphRules(dataset_params, partition='test')
    val_loader = DataLoader(val_dataset, batch_size=trainer_params.batch_size, shuffle=False)

    # Define model and initialize weights
    if base_params.gr_single_relationship:
        row_model = GraphRulesSingleRelationship(base_params)
        col_model = GraphRulesSingleRelationship(base_params)
        row_model.apply(weights_init)
        col_model.to(DEVICE)
        col_model.apply(weights_init)
        row_model.to(DEVICE)
    elif base_params.gr_multi_label:
        model = GraphRulesMultiLabel(base_params)
        model.to(DEVICE)
        model.apply(weights_init)
    elif base_params.gr_multi_task:
        model = GraphRulesMultiTask(base_params)
        model.to(DEVICE)
        model.apply(weights_init)

    # Define optimizer and learning rate scheduler
    if trainer_params.optimizer == 'adam' and not base_params.gr_single_relationship:
        optimizer = optim.Adam(model.parameters(), lr=trainer_params.lr,
                                betas=(trainer_params.beta1, trainer_params.beta2))
    elif trainer_params.optimizer == 'adam' and base_params.gr_single_relationship:
        row_optimizer = optim.Adam(row_model.parameters(), lr=trainer_params.lr,
                                betas=(trainer_params.beta1, trainer_params.beta2))
        col_optimizer = optim.Adam(col_model.parameters(), lr=trainer_params.lr,
                                betas=(trainer_params.beta1, trainer_params.beta2))

    if trainer_params.schedule_lr and not base_params.gr_single_relationship:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=trainer_params.lr_patience,
                                            factor=trainer_params.lr_reduce_factor, verbose=True, 
                                            mode=trainer_params.lr_schedule_mode,
                                            cooldown=trainer_params.lr_cooldown, min_lr=trainer_params.min_lr)
    elif trainer_params.optimizer == 'adam' and base_params.gr_single_relationship:
        row_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(row_optimizer, patience=trainer_params.lr_patience,
                                            factor=trainer_params.lr_reduce_factor, verbose=True, 
                                            mode=trainer_params.lr_schedule_mode,
                                            cooldown=trainer_params.lr_cooldown, min_lr=trainer_params.min_lr)
        col_lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(col_optimizer, patience=trainer_params.lr_patience,
                                            factor=trainer_params.lr_reduce_factor, verbose=True,
                                            mode=trainer_params.lr_schedule_mode,
                                            cooldown=trainer_params.lr_cooldown, min_lr=trainer_params.min_lr)
    
    # Define loss criteria
    if trainer_params.loss_criteria == 'nll':
        loss_criteria = nn.NLLLoss()
    elif trainer_params.loss_criteria == 'bce_logits':
        weight_tensor = torch.Tensor([5]).to(DEVICE)
        loss_criteria = nn. BCEWithLogitsLoss(pos_weight=weight_tensor)

    # Watch model
    # if base_params.gr_single_relationship:
    #     wandb.watch(row_model)
    #     wandb.watch(col_model)
    # else:
    #     wandb.watch(model)


    # Model saving params
    best_accuracy = 0.0
    best_epoch = -1

    # Outer training loop
    print("$$$ STARTING TRAINING LOOP $$$")
    for epoch in range(trainer_params.num_epochs.num_epochs):
        with trange(len(train_loader)) as t:
            # Train a single pass of the model
            if base_params.gr_single_relationship:
                train_out_dict = train_sr(row_model, col_model, row_optimizer, col_optimizer, train_loader, loss_criteria)   
            elif base_params.gr_multi_label:
                train_out_dict = train_ml(model, optimizer, train_loader, loss_criteria)
            elif base_params.gr_multi_task:
                train_out_dict = train_mt(model, optimizer, train_loader, loss_criteria)
            
            # Log training info
            wandb.log(train_out_dict)

            # Perform evaluation at intervals
            if epoch % trainer_params.val_interval == 0:
                if base_params.gr_single_relationship:
                    eval_out_dict = eval_sr(row_model, col_model, val_loader, loss_criteria)
                elif base_params.gr_multi_label:
                    eval_out_dict = eval_ml(model, val_loader, loss_criteria)
                elif base_params.gr_multi_task:
                    eval_out_dict = eval_mt(model, val_loader, loss_criteria)
                
                wandb.log(eval_out_dict)

            # Schedule learning rate
            if trainer_params.schedule_lr:
                if base_params.gr_single_relationship:
                    row_lr_scheduler.step(train_out_dict['row_loss'])
                    col_lr_scheduler.step(train_out_dict['col_loss'])
                else:
                    lr_scheduler.step(train_out_dict['train_loss'])

            # Save models based on accuracy/F1 score
            if epoch % trainer_params.val_interval == 0:
                if eval_out_dict['val_acc'] > best_accuracy:
                    if base_params.gr_single_relationship:
                        row_model_path = log_path + os.sep + "row_only_net_{}_best_so_far.pth".format(epoch)
                        col_model_path = log_path + os.sep + "col_only_net_{}_best_sor_far.pth".format(epoch)
                        torch.save(row_model.state_dict(), row_model_path)
                        torch.save(col_model.state_dict(), col_model_path)
                    else:
                        model_path = log_path + os.sep + "net_{}_best_so_far.pth".format(epoch)
                        torch.save(model.state_dict(), model_path)
                    best_accuracy = eval_out_dict['val_acc']
                    best_epoch = epoch

            t_postfix_dict = {
                'train_loss': train_out_dict['train_loss'],
                'val_loss': eval_out_dict['val_loss']
                'val_acc': eval_out_dict['val_acc']
            }
            t.set_postfix(t_postfix_dict)
            t.update()

    print("*** TRAINING IS COMPLETE ***")
    print("Best validation accuracy: {}, in epoch: {}".format(best_accuracy, best_epoch))


def train_sr(row_model, col_model, row_optimizer, col_optimizer, train_loader, loss_criteria):
    # Set model to train mode
    row_model.train()
    col_model.train()


def train_ml(model, optimizer, train_loader, loss_criteria):
    """
    Multi-label: Model produces all of the labels together
    """
    # Set model to train()
    model.train()

    epoch_loss = 0.0

    # Train loop for a single pass on the dataset
    for idx, data in enumerate(train_loader):
        # Perform single train step
        optimizer.zero_grad()
        data = data.to(DEVICE)


def train_mt(model, optimizer, train_loader, loss_criteria):
    """
    Multi-task: Model produces separate row and col predictions
    """
    pass


def eval_sr(row_model, col_model, val_loader, loss_criteria):
    pass


def eval_ml(model, val_loader, loss_criteria):
    pass


def eval_mt(model, val_loader, loss_criteria):
    pass


if __name__ == '__main__':
    # Get argument dictionaries
    base_params = base_params()
    dataset_params = dataset_params()

    # Seed things
    torch.manual_seed(trainer_params.seed)
    torch.cuda.manual_seed(trainer_params.seed)
    np.random.seed(trainer_params.seed)
    random.seed(trainer_params.seed)

    # Create save locations
    time = datetime.now()
    time = time.strftime('%Y_%m_%d_%H_%M')
    root_path = os.getcwd() + os.sep + trainer_params.exp + os.sep + trainer_params.run + '_' + time
    mkdir_p(root_path)
    log_path = root_path + os.sep + '/checkpoints'
    mkdir_p(log_path)

    # Set CUDA access
    use_cuda = torch.cuda.is_available() and trainer_params.device == 'cuda'
    if use_cuda:
        DEVICE = torch.device("cuda")
    else:
        print('Warning: CPU is being used to run model,\
             CUDA device not being used for current run')
        DEVICE = torch.device("cpu")

    # Create wandb config dict
    config_dict = {'base_params': vars(base_params),
                    'dataset_params': vars(dataset_params)
                  }
    
    print("#" * 100)
    print("CURRENT CONFIGURATION")
    print("#" * 100)
    print(config_dict)
    print("#" * 100)

    # Initialize wandb config
    # wandb_name = trainer_params.exp + '_' + trainer_params.run + '_' + time
    # wandb.init(name=wandb_name, entity='rsaha', project='table_graph_rules', config=config_dict)

    namespace_config_dict = {
                                'dataset_params': dataset_params,
                                'base_params': base_params
                            }
                            
