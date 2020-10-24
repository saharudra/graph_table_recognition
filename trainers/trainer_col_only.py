from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data 
from torch_geometric.data import Dataset, DataLoader
from torch_scatter import scatter_mean
import torch_geometric.transforms as GT

import numpy as np
import os
import math
import json
import random
import wandb

from models.version1 import TbNetV1
from dataloaders.scitsr import ScitsrDataset
from misc.args import train_params, scitsr_params, img_model_params, base_params
from ops.misc import weights_init, mkdir_p


def main(config):
    # Separate params
    dataset_params = config['dataset_params']
    img_model_params = config['img_params']
    base_params = config['model_base_params']
    trainer_params = config['trainer_params']

    # Define dataloader
    train_dataset = ScitsrDataset(dataset_params, partition='train')
    train_loader = DataLoader(train_dataset, batch_size=trainer_params.batch_size, shuffle=True)

    val_dataset = ScitsrDataset(dataset_params, partition='test')
    val_loader = DataLoader(val_dataset, batch_size=trainer_params.batch_size, shuffle=False)

    # Define model
    model = TbNetV1(base_params, img_model_params, trainer_params)
    model.to(DEVICE)
    # TODO: Weight initialization
    # model.apply(weights_init)

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
    loss_criteria = nn.NLLLoss()

    # Watch model
    wandb.watch(model)

    # Model saving params
    best_accuracy = 0.0
    best_epoch = -1

    # Outer train loop
    print("*** STARTING TRAINING LOOP ***")
    for epoch in range(trainer_params.num_epochs):
        train_loss = train(model, optimizer, train_loader, epoch, loss_criteria)
        
        # Log information
        wandb.log(
            {
                'train_loss': train_loss
            }
        )

        if epoch % trainer_params.val_interval == 0:
            val_loss, val_acc= eval(model, val_loader, epoch, loss_criteria)

            # Log information
            wandb.log(
                {
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_acc': val_acc
                }
            )

        # Schedule learning rate
        if trainer_params.schedule_lr:
            lr_scheduler.step(train_loss)
        
        # Save models based on accuracy
        if epoch % trainer_params.save_interval == 0:
            if val_acc > best_accuracy:
                model_path = log_path + os.sep + "net_{}_best_so_far.pth".format(epoch)
                torch.save(model.state_dict(), model_path)
                best_accuracy = val_acc
                best_epoch = epoch
        
    print("*** TRAINING IS COMPLETE ***")
    print("Best validation accuracy: {}, in epoch: {}".format(best_accuracy, best_epoch))


def train(model, optimizer, train_loader, epoch, loss_criteria):
    # Train loop for a batch
    model.train()

    epoch_loss = 0.0

    for idx, data in enumerate(train_loader):
        # Perform single train step
        data = data.to(DEVICE)
        col_pred = model(data)
        col_loss = loss_function(col_pred, data.y_col, loss_criteria)
        # Overall loss
        loss = col_loss
        loss.backward()
        optimizer.step()
        
        # Aggregate losses
        epoch_loss += loss.item()

    epoch_loss /= len(train_loader.dataset)
    print("Epoch: {}, Overall Loss: {}".format(epoch, epoch_loss))

    return epoch_loss


def eval(model, val_loader, epoch, loss_criteria):
    # Eval loop 
    model.eval()
    
    val_loss = 0.0

    n_correct_col = 0.0
    n_total_col = 0.0

    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            data = data.to(DEVICE)
            col_pred = model(data)
            col_loss = loss_function(col_pred, data.y_col, loss_criteria)
            loss = col_loss

            val_loss += loss.item()
            
            # Accuracy calculation
            _, col_pred = col_pred.max(1)
            
            col_label = data.y_col.detach().cpu().numpy()
            col_pred = col_pred.detach().cpu().numpy()

            n_correct_col = n_correct_col + (col_label == col_pred).sum()
            n_total_col += col_label.shape[0]

    val_loss /= len(val_loader.dataset)
    val_acc = n_correct_col / n_total_col

    return val_loss, val_acc


def loss_function(pred_col, data_col, loss_criteria):

    col_loss = loss_criteria(pred_col, data_col)

    return col_loss


if __name__ == "__main__":
    # Get argument dictionaries
    img_params = img_model_params()
    dataset_params = scitsr_params()
    trainer_params = train_params()
    model_base_params = base_params()

    # Set params for row only things
    trainer_params.row_only = False
    trainer_params.col_only = True
    trainer_params.multi_task = False

    # Seed things
    torch.manual_seed(trainer_params.seed)
    torch.cuda.manual_seed(trainer_params.seed)
    np.random.seed(trainer_params.seed)
    random.seed(trainer_params.seed)

    # Create save locations
    root_path = os.getcwd() + os.sep + trainer_params.exp + os.sep + trainer_params.run
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

    namespace_config_dict = {'img_params': img_params,
                             'dataset_params': dataset_params,
                             'trainer_params': trainer_params,
                             'model_base_params': model_base_params
                            }

    main(config=namespace_config_dict)

    