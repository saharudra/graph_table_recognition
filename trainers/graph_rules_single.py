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
from misc.args import scitsr_params, base_params, trainer_params, img_model_params
from ops.utils import cal_adj_label
from ops.misc import weights_init, mkdir_p


def main(config):
    dataset_params = config['dataset_params']
    base_params = config['base_params']
    trainer_params = config['trainer_params']
    img_model_params = config['img_model_params']

    # Define dataloader
    train_dataset = ScitsrGraphRules(dataset_params, partition='train')
    train_loader = DataLoader(train_dataset, batch_size=trainer_params.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True)
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # Uncomment when testing new functionality

    val_dataset = ScitsrGraphRules(dataset_params, partition='test')
    # Using batch size of 1 for validation with adjacency matrix
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)
    
    # Define model and initialize weights
    if dataset_params.gr_single_relationship:
        model = GraphRulesSingleRelationship(base_params, img_model_params)
        model.apply(weights_init)
        model.to(DEVICE)

    if trainer_params.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=trainer_params.lr,
                                betas=(trainer_params.beta1, trainer_params.beta2))

    if trainer_params.schedule_lr:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=trainer_params.lr_patience,
                                            factor=trainer_params.lr_reduce_factor, verbose=True, 
                                            mode=trainer_params.lr_schedule_mode,
                                            cooldown=trainer_params.lr_cooldown, min_lr=trainer_params.min_lr)

    # Define loss criteria
    if trainer_params.loss_criteria == 'nll':
        loss_criteria = nn.NLLLoss()
    elif trainer_params.loss_criteria == 'bce_logits':
        weight_tensor = torch.Tensor([trainer_params.class_weight]).to(DEVICE)
        loss_criteria = nn. BCEWithLogitsLoss(pos_weight=weight_tensor, reduction='sum')

    # Watch model
    # wandb.watch(model)

    # Model saving params
    best_accuracy = 0.0
    best_epoch = -1

    if trainer_params.overfit_one_batch:
        print("$$$ OVERFITTING ON ONE BATCH")
        data = next(iter(train_loader))
        with trange(trainer_params.num_epochs) as t:
            for epoch in t:
                train_out_dict = train_sr_overfit_one_batch(model, optimizer, data, loss_criteria, trainer_params)
                
                # wandb.log(train_out_dict)
                t.set_postfix(train_out_dict)
                t.update()

    else:
        # Outer training loop
        print('LENGTH OF TRAINING DATASET: {}, VALIDATION DATASET: {}'.format(len(train_loader), len(val_loader)))
        print("$$$ STARTING TRAINING LOOP $$$")
        with trange(trainer_params.num_epochs) as t:
            for epoch in t:
                # Train a single pass of the model
                if dataset_params.gr_single_relationship:
                    train_out_dict = train_sr(model, optimizer, train_loader, loss_criteria, trainer_params)   
                    print(train_out_dict)
                elif dataset_params.gr_multi_label:
                    train_out_dict = train_ml(model, optimizer, train_loader, loss_criteria, trainer_params)
                elif dataset_params.gr_multi_task:
                    train_out_dict = train_mt(model, optimizer, train_loader, loss_criteria, trainer_params)
                
                # Log training info
                # wandb.log(train_out_dict)

                # Perform evaluation at intervals
                if epoch % trainer_params.val_interval == 0:
                    if dataset_params.gr_single_relationship:
                        eval_out_dict = eval_sr(model, val_loader, loss_criteria, trainer_params)
                        print(eval_out_dict)
                    elif dataset_params.gr_multi_label:
                        eval_out_dict = eval_ml(model, val_loader, loss_criteria, trainer_params)
                    elif dataset_params.gr_multi_task:
                        eval_out_dict = eval_mt(model, val_loader, loss_criteria, trainer_params)
                    
                    # wandb.log(eval_out_dict)

                # Schedule learning rate
                if trainer_params.schedule_lr:
                        lr_scheduler.step(train_out_dict['train_loss'])

                # Save models based on accuracy/F1 score
                if epoch % trainer_params.val_interval == 0:
                    if eval_out_dict['val_acc'] > best_accuracy:
                        if dataset_params.gr_single_relationship:
                            if trainer_params.row_only == True:
                                model_path = log_path + os.sep + "row_only_net_{}_best_so_far.pth".format(epoch)
                            elif trainer_params.col_only == True:
                                model_path = log_path + os.sep + "col_only_net_{}_best_so_far.pth".format(epoch)
                            torch.save(model.state_dict(), model_path)
                        best_accuracy = eval_out_dict['val_acc']
                        best_epoch = epoch

                t_postfix_dict = {
                    'val_acc': eval_out_dict['val_acc'],
                }
                t.set_postfix(t_postfix_dict)
                t.update()

        print("*** TRAINING IS COMPLETE ***")
        print("Best validation accuracy: {}, in epoch: {}".format(best_accuracy, best_epoch))

def train_sr_overfit_one_batch(model, optimizer, data, loss_criteria, trainer_params):
    optimizer.zero_grad()
    if trainer_params.row_only == True:
        row_data, col_data = data
        data = row_data
    elif trainer_params.col_only == True:
        row_data, col_data = data
        data = col_data
    data = data.to(DEVICE)

    # Gather model output and convert into a row tensor same as the ground truth
    logits = model(data)

    # Compute individual model losses
    batch_loss = loss_function(logits, data.y, loss_criteria, task='sr')

    # Update models and optimizers
    batch_loss.backward()
    optimizer.step()

    # Aggregate losses
    loss = batch_loss.item() / len(data.y)


    # Calculate accuracy
    _, pred = logits.max(1)

    # For sanity losses of all zero and all one predictions
    all_one = torch.ones(pred.shape).numpy()
    all_zeros = torch.zeros(pred.shape).numpy()

    
    label = data.y.detach().cpu().numpy()
    
    pred = pred.detach().cpu().numpy()
    
    n_correct = (label == pred).sum()
    n_correct_one = (label == all_one).sum()
    n_correct_zero = (label == all_zeros).sum()
    n_total_row = label.shape[0]

    acc = n_correct / n_total
    all_one_acc = n_correct_one / n_total
    all_zero_acc = n_correct_zero / n_total

    out_dict = {
        'train_loss': loss,
        'train_acc': acc,
        'all_one_acc': all_one_acc,
        'all_zero_acc': all_zero_acc,
    }

    return out_dict


def train_sr(model, optimizer, train_loader, loss_criteria, trainer_params):
    """
    Single Relationship: Separate models trained for row/col relationships
    """
    # Set model to train mode
    model.train()

    # Initialize losses to be monitored
    train_loss = 0.0

    for idx, data in enumerate(train_loader):
        # Perform single step of training
        optimizer.zero_grad()
        if trainer_params.row_only == True:
            data, _ = data
        elif trainer_params.col_only == True:
            _, data = data
        data = data.to(DEVICE)


        # Gather model output and convert into a row tensor same as the ground truth
        logits = model(data)

        # Compute model losses
        batch_loss = loss_function(logits, data.y, loss_criteria, task='sr')

        # Update models and optimizers
        batch_loss.backward()
        optimizer.step()

        # Aggregate losses
        train_loss += batch_loss.detach().item()

    train_loss /= len(train_loader.dataset)

    out_dict = {'train_loss': train_loss}

    return out_dict


def train_ml(model, optimizer, train_loader, loss_criteria, trainer_params):
    """
    Multi-label: Model produces all of the labels together
    """
    raise NotImplementedError
        

def train_mt(model, optimizer, train_loader, loss_criteria, trainer_params):
    """
    Multi-task: Model produces separate row and col predictions
    """
    raise NotImplementedError


def eval_sr(model, val_loader, loss_criteria, trainer_params):
    """
    Evaluation of single relationship models
    """
    model.eval()

    val_loss = 0.0
    val_acc = 0.0
    n_correct = 0.0
    n_total = 0.0

    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            if trainer_params.row_only == True:
                data, _ = data
            elif trainer_params.col_only == True:
                _, data = data
            data = data.to(DEVICE)

            logits = model(data)
            batch_loss = loss_function(logits, data.y, loss_criteria, task='sr')

            val_loss += batch_loss.detach().item()

            _, pred = logits.max(1)

            label = data.y.detach().cpu().numpy()
            pred = pred.detach().cpu().numpy()

            n_correct = n_correct + (label == pred).sum()
            n_total += label.shape[0]

            # Calculate Precision, Recall and F1 Score for cell adjacency

        val_loss /= len(val_loader.dataset)
        val_acc = n_correct / n_total
        
        out_dict = {
            'val_loss': val_loss,
            'val_acc': val_acc
        }
    
    return out_dict


def loss_function(logits, gt, loss_criteria, task):
    if task == 'sr':
        # Single Relationship loss calculation
        loss = loss_criteria(logits, gt)

    elif task == 'ml':
        # Multi-label loss calculation
        raise NotImplementedError

    elif task == 'mt':
        # Multi-task loss calculation
        raise NotImplementedError

    return loss


if __name__ == '__main__':
    # Get argument dictionaries
    base_params = base_params()
    dataset_params = scitsr_params()
    img_model_params = img_model_params()
    trainer_params = trainer_params()

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
    # wandb_name = trainer_params.run + '_' + time
    # wandb.init(name=wandb_name, entity='rsaha', project='table_graph_rules', config=config_dict)

    namespace_config_dict = {
                                'dataset_params': dataset_params,
                                'base_params': base_params,
                                'trainer_params': trainer_params, 
                                'img_model_params': img_model_params
                            }

    main(config=namespace_config_dict)