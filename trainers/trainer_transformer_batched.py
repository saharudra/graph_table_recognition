"""
Sequences padded to a max length to create batches before being fed
to the model. The dataloader is different and access to tesor is varied.
Copying `./trainers/trainer_transformer.py` here and changing dataloader
access.
"""
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data 
from torch.utils.data import Dataset, DataLoader

import numpy as np
import os
import math
import json
import random
import wandb
from datetime import datetime
import tqdm

from models.transformer import TbTSR
from dataloaders.scitsr_loader import ScitsrDatasetSB
from dataloaders.icdar_transformer import Icdar2013DatasetSB, Icdar2019DatasetSB
from dataloaders.pubtabnet import PubTabNetDataset
from misc.args import *
from ops.misc import weights_init, mkdir_p


def main(config):
    # Separate params
    dataset_params = config['dataset_params']
    img_model_params = config['img_params']
    base_params = config['model_params']
    trainer_params = config['trainer_params']

    # Define dataloaders
    if trainer_params.dataset == 'scitsr':
        train_dataset = ScitsrDatasetSB(dataset_params)
        train_loader = DataLoader(train_dataset, batch_size=trainer_params.batch_size, shuffle=True)

        # val_dataset = ScitsrDatasetSB(dataset_params, partition='test')
        # val_loader = DataLoader(val_dataset, batch_size=trainer_params.batch_size, shuffle=False)
        # TODO: Remove this for full training
        val_loader = train_loader

    elif trainer_params.dataset == 'pubtabnet':
        train_dataset = PubTabNetDataset(dataset_params)
        train_loader = DataLoader(train_dataset, batch_size=trainer_params.batch_size, shuffle=True)

        val_dataset = PubTabNetDataset(dataset_params, partition='val')
        val_loader = DataLoader(val_dataset, batch_size=trainer_params.batch_size, shuffle=False)

        test_dataset = PubTabNetDataset(dataset_params, partition='test')
        test_loader = DataLoader(test_dataset, batch_size=trainer_params.batch_size, shuffle=False)

    if trainer_params.eval_dataset == 'icdar2013':
        eval_dataset = Icdar2013DatasetSB(dataset_params)
        eval_loader = DataLoader(eval_dataset, batch_size=trainer_params.batch_size, shuffle=False)

    elif trainer_params.eval_dataset == 'icdar2019':
        eval_dataset = Icdar2019DatasetSB(dataset_params)
        eval_loader = DataLoader(eval_dataset, batch_size=trainer_params.batch_size, shuffle=False)

    # Define model
    model = TbTSR(base_params, img_model_params, trainer_params)
    model.to(DEVICE)
    model.apply(weights_init)

    # Define optimizer and learning rate scheduler
    if trainer_params.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=trainer_params.lr,
                                betas=(trainer_params.beta1, trainer_params.beta2))
    elif trainer_params.optimizer == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=trainer_params.lr)
    elif trainer_params.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr=trainer_params.lr)

    if trainer_params.schedule_lr:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=trainer_params.lr_patience,
                                            factor=trainer_params.lr_reduce_factor, verbose=True, 
                                            mode=trainer_params.lr_schedule_mode,
                                            cooldown=trainer_params.lr_cooldown, min_lr=trainer_params.min_lr)

    # Define loss criteria
    # TODO: Try cross entropy loss with weight tensor
    weight_tensor = torch.Tensor([50]).to(DEVICE)
    loss_criteria = nn.BCEWithLogitsLoss(pos_weight=weight_tensor)

    # Watch model
    wandb.watch(model)

    # Model saving params
    best_accuracy = 0.0
    best_epoch = -1

    # Outer train loop
    print("*** STARTING TRAINING LOOP ***")
    print("Length of training dataset: {}".format(len(train_loader)))
    print("Length of validation dataset: {}".format(len(val_loader)))
    for epoch in range(trainer_params.num_epochs):
        train_out_dict = train(model, optimizer, train_loader, loss_criteria, trainer_params)
        print("Epoch: {}, Overall Loss: {}".format(epoch, train_out_dict['train_loss']))
        
        # Log information
        wandb.log(train_out_dict)

        if epoch % trainer_params.val_interval == 0:
            val_out_dict = eval(model, val_loader, loss_criteria, trainer_params)

            # Log information
            wandb.log(val_out_dict)
            print('Validation Accuracy: {}, Validation Row Acc: {}, Validation Col Acc: {}'.format(val_out_dict['val_acc'],
                                                                                                   val_out_dict['val_row_acc'],
                                                                                                   val_out_dict['val_col_acc']))
        
        # Schedule learning rate
        if trainer_params.schedule_lr:
            lr_scheduler.step(val_out_dict['val_loss'])
        
        # Save models based on accuracy
        if epoch % trainer_params.save_interval == 0:
            if val_out_dict['val_acc'] > best_accuracy:
                model_path = log_path + os.sep + "net_{}_best_so_far.pth".format(epoch)
                torch.save(model.state_dict(), model_path)
                best_accuracy = val_out_dict['val_acc']
                best_epoch = epoch
        
    print("*** TRAINING IS COMPLETE ***")
    print("Best validation accuracy: {}, in epoch: {}".format(best_accuracy, best_epoch))


def train(model, optimizer, train_loader, loss_criteria, trainer_params):
    # Train loop for a batch
    model.train()

    epoch_loss = 0.0
    row_loss = 0.0
    col_loss = 0.0

    for idx, data in enumerate(train_loader):
        # Perform single train step
        for key, value in data.items():
                data[key] = data[key].to(DEVICE)

        row_pred, col_pred = model(data)
        row_loss = loss_function(row_pred.view(trainer_params.batch_size, -1), data['y_row'], loss_criteria)
        col_loss = loss_function(col_pred.view(trainer_params.batch_size, -1), data['y_col'], loss_criteria)
        # Overall loss
        loss = row_loss + col_loss
        loss.backward()

        # Accumulate gradients for training stability
        # if (idx + 1) % trainer_params.optimizer_accu_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        
        # Aggregate losses
        epoch_loss += loss.item()
        row_loss += row_loss.item()
        col_loss += col_loss.item()
    
    epoch_loss /= len(train_loader.dataset)
    row_loss /= len(train_loader.dataset)
    col_loss /= len(train_loader.dataset)

    train_out_dict = {  
        'train_loss': epoch_loss,
        'train_row_loss': row_loss,
        'train_col_loss': col_loss
    }

    return train_out_dict


def eval(model, val_loader, loss_criteria, trainer_params):
    # Eval loop 
    model.eval()

    val_loss = 0.0
    row_loss = 0.0
    col_loss = 0.0

    n_correct_row = 0.0
    n_correct_col = 0.0
    n_total_row = 0.0
    n_total_col = 0.0

    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            for key, value in data.items():
                data[key] = data[key].to(DEVICE)
            
            row_pred, col_pred = model(data)
            row_pred, col_pred = row_pred.view(trainer_params.batch_size, -1), col_pred.view(trainer_params.batch_size, -1)
            # import pdb; pdb.set_trace()
            row_loss = loss_function(row_pred, data['y_row'], loss_criteria)
            col_loss = loss_function(col_pred, data['y_col'], loss_criteria)
            loss = row_loss + col_loss

            val_loss += loss.item()
            row_loss += row_loss.item()
            col_loss += col_loss.item()

            # Accuracy calculation
            # _, row_pred = F.softmax(row_pred).max(1)
            # _, col_pred = F.softmax(col_pred).max(1)

            row_label = data['y_row'].detach().cpu().numpy()
            col_label = data['y_col'].detach().cpu().numpy()
            # row_pred = row_pred.detach().cpu().numpy()
            # col_pred = col_pred.detach().cpu().numpy()

            # n_correct_row = n_correct_row + (row_label == row_pred).sum()
            # n_correct_col = n_correct_col + (col_label == col_pred).sum()
            n_total_row += row_label.shape[0]
            n_total_col += col_label.shape[0]

    val_loss /= len(val_loader.dataset)
    row_loss /= len(val_loader.dataset)
    col_loss /= len(val_loader.dataset)

    row_acc = n_correct_row / n_total_row
    col_acc = n_correct_col / n_total_col
    val_acc = (row_acc + col_acc) * 0.5

    val_out_dict = {
        'val_loss': val_loss,
        'val_row_loss': row_loss,
        'val_col_loss': col_loss,
        'val_acc': val_acc,
        'val_row_acc': row_acc,
        'val_col_acc': col_acc
    }

    return val_out_dict


def loss_function(pred, target, loss_criteria):
    
    loss = loss_criteria(pred, target)

    return loss


if __name__ == '__main__':
    # Get argument dictionaries
    img_params = img_model_params()
    dataset_params = scitsr_params()
    trainer_params = trainer_params()
    model_params = base_params()

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
        print("Using CUDA resources for training")
        DEVICE = torch.device("cuda")
    else:
        print('Warning: CPU is being used to run model,\
             CUDA device not being used for current run')
        DEVICE = torch.device("cpu")

    # Create wandb config dict
    config_dict = {'img_params': vars(img_params),
                    'dataset_params': vars(dataset_params),
                    'trainer_params': vars(trainer_params),
                    'model_base_params': vars(model_params)}
    print("#" * 100)
    print("CURRENT CONFIGURATION")
    print("#" * 100)
    print(config_dict)
    print("#" * 100)

    # Initialize wandb config
    wandb_name = trainer_params.run + '_' + time
    wandb.init(name=wandb_name, entity='rsaha', project='table_structure_recognition', config=config_dict)

    namespace_config_dict = {'img_params': img_params,
                             'dataset_params': dataset_params,
                             'trainer_params': trainer_params,
                             'model_params': model_params
                            }

    main(config=namespace_config_dict)