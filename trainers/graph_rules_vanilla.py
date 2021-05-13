from __future__ import print_function
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data 
from torch_geometric.data import DataLoader

import numpy as np
import os
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
    # TODO: Collate adjacency matrix to pass it as a batch
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, pin_memory=True)
    
    # Define model and initialize weights
    if dataset_params.gr_single_relationship:
        row_model = GraphRulesSingleRelationship(base_params, img_model_params)
        col_model = GraphRulesSingleRelationship(base_params, img_model_params)
        row_model.apply(weights_init)
        # row_model = nn.DataParallel(row_model)
        col_model.apply(weights_init)
        # col_model = nn.DataParallel(col_model)
        col_model.to(DEVICE)
        row_model.to(DEVICE)

    # Define optimizer and learning rate scheduler
    if trainer_params.optimizer == 'adam' and not dataset_params.gr_single_relationship:
        optimizer = optim.Adam(model.parameters(), lr=trainer_params.lr,
                                betas=(trainer_params.beta1, trainer_params.beta2))
    elif trainer_params.optimizer == 'adam' and dataset_params.gr_single_relationship:
        row_optimizer = optim.Adam(row_model.parameters(), lr=trainer_params.lr,
                                betas=(trainer_params.beta1, trainer_params.beta2))
        col_optimizer = optim.Adam(col_model.parameters(), lr=trainer_params.lr,
                                betas=(trainer_params.beta1, trainer_params.beta2))

    if trainer_params.schedule_lr and not dataset_params.gr_single_relationship:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=trainer_params.lr_patience,
                                            factor=trainer_params.lr_reduce_factor, verbose=True, 
                                            mode=trainer_params.lr_schedule_mode,
                                            cooldown=trainer_params.lr_cooldown, min_lr=trainer_params.min_lr)
    elif trainer_params.optimizer == 'adam' and dataset_params.gr_single_relationship:
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
        weight_tensor = torch.Tensor([trainer_params.class_weight]).to(DEVICE)
        loss_criteria = nn. BCEWithLogitsLoss(pos_weight=weight_tensor, reduction='sum')

    # Watch model
    if dataset_params.gr_single_relationship:
        wandb.watch(row_model)
        wandb.watch(col_model)
    else:
        wandb.watch(model)


    # Model saving params
    best_accuracy = 0.0
    best_epoch = -1

    if trainer_params.overfit_one_batch:
        print("$$$ OVERFITTING ON ONE BATCH $$$")
        data = next(iter(train_loader))
        with trange(trainer_params.num_epochs) as t:
            for epoch in t:
                if dataset_params.gr_single_relationship:
                    train_out_dict = train_sr_overfit_one_batch(row_model, col_model, row_optimizer, col_optimizer, data, loss_criteria)
                
                wandb.log(train_out_dict)
                t.set_postfix(train_out_dict)
                t.update()
    else:
        # Outer training loop
        print('LENGTH OF TRAINING DATASET: {}, VALIDATION DATASET: {}'.format(len(train_loader.dataset), len(val_loader.dataset)))
        print("$$$ STARTING TRAINING LOOP $$$")
        with trange(trainer_params.num_epochs) as t:
            for epoch in t:
                # Train a single pass of the model
                if dataset_params.gr_single_relationship:
                    train_out_dict = train_sr(row_model, col_model, row_optimizer, col_optimizer, train_loader, loss_criteria)                
                
                # Log training info
                wandb.log(train_out_dict)

                # Perform evaluation at intervals
                if epoch % trainer_params.val_interval == 0:
                    if dataset_params.gr_single_relationship:
                        eval_out_dict = eval_sr(row_model, col_model, val_loader, loss_criteria)
                    wandb.log(eval_out_dict)

                # Schedule learning rate
                if trainer_params.schedule_lr:
                    if dataset_params.gr_single_relationship:
                        row_lr_scheduler.step(train_out_dict['row_loss'])
                        col_lr_scheduler.step(train_out_dict['col_loss'])
                    else:
                        lr_scheduler.step(train_out_dict['train_loss'])

                # Save models based on accuracy/F1 score
                if epoch % trainer_params.val_interval == 0:
                    if eval_out_dict['val_acc'] > best_accuracy:
                        if dataset_params.gr_single_relationship:
                            row_model_path = log_path + os.sep + "row_only_net_{}_best_so_far.pth".format(epoch)
                            col_model_path = log_path + os.sep + "col_only_net_{}_best_so_far.pth".format(epoch)
                            torch.save(row_model.state_dict(), row_model_path)
                            torch.save(col_model.state_dict(), col_model_path)
                        else:
                            model_path = log_path + os.sep + "net_{}_best_so_far.pth".format(epoch)
                            torch.save(model.state_dict(), model_path)
                        best_accuracy = eval_out_dict['val_acc']
                        best_epoch = epoch

                t_postfix_dict = {
                    'val_acc': eval_out_dict['val_acc'],
                    'val_f1': eval_out_dict['val_f1'],
                }
                t.set_postfix(t_postfix_dict)
                t.update()

        print("*** TRAINING IS COMPLETE ***")
        print("Best validation accuracy: {}, in epoch: {}".format(best_accuracy, best_epoch))


def train_sr_overfit_one_batch(row_model, col_model, row_optimizer, col_optimizer, data, loss_criteria):
    row_optimizer.zero_grad()
    col_optimizer.zero_grad()
    row_data, col_data = data
    row_data = row_data.to(DEVICE)
    col_data = col_data.to(DEVICE)

    # Gather model output and convert into a row tensor same as the ground truth
    row_logits = row_model(row_data)
    col_logits = col_model(col_data)

    # Compute individual model losses
    batch_row_loss = loss_function(row_logits, row_data.y, loss_criteria, task='sr')
    batch_col_loss = loss_function(col_logits, col_data.y, loss_criteria, task='sr')

    # Update models and optimizers
    batch_row_loss.backward()
    row_optimizer.step()

    batch_col_loss.backward()
    col_optimizer.step()

    # Aggregate losses
    row_loss = batch_row_loss.item() / len(row_data.y)
    col_loss = batch_col_loss.item() / len(col_data.y)
    train_loss = (row_loss + col_loss) * 0.5
    # import pdb; pdb.set_trace()

    # Calculate accuracy
    # import pdb; pdb.set_trace()
    _, row_pred = row_logits.max(1)
    _, col_pred = col_logits.max(1)

    # For sanity losses of all zero and all one predictions
    all_one_row = torch.ones(row_pred.shape).numpy()
    all_zeros_row = torch.zeros(row_pred.shape).numpy()
    all_one_col = torch.ones(col_pred.shape).numpy()
    all_zeros_col = torch.zeros(col_pred.shape).numpy()
    # import pdb; pdb.set_trace()

    
    row_label = row_data.y.detach().cpu().numpy()
    col_label = col_data.y.detach().cpu().numpy()

    row_pred = row_pred.detach().cpu().numpy()
    col_pred = col_pred.detach().cpu().numpy()

    
    n_correct_row = (row_label == row_pred).sum()
    n_correct_col = (col_label == col_pred).sum()
    n_correct_row_one = (row_label == all_one_row).sum()
    n_correct_row_zero = (row_label == all_zeros_row).sum()
    n_correct_col_one = (col_label == all_one_col).sum()
    n_correct_col_zero = (col_label == all_zeros_col).sum()
    n_total_row = row_label.shape[0]
    n_total_col = col_label.shape[0]

    row_acc = n_correct_row / n_total_row
    col_acc = n_correct_col / n_total_col
    row_one = n_correct_row_one / n_total_row
    row_zero = n_correct_row_zero / n_total_row
    col_one = n_correct_col_one / n_total_col
    col_zero = n_correct_col_zero / n_total_col
    train_acc = 0.5 * (row_acc + col_acc)

    out_dict = {
        'train_loss': train_loss,
        'row_loss': row_loss,
        'col_loss': col_loss,
        'train_acc': train_acc,
        'row_acc': row_acc,
        'col_acc': col_acc,
        # 'row_all_one_acc': row_one,
        # 'row_all_zero_acc': row_zero,
        # 'col_all_one_acc': col_one,
        # 'col_all_zero_acc': col_zero
    }

    return out_dict


def train_sr(row_model, col_model, row_optimizer, col_optimizer, train_loader, loss_criteria):
    """
    Single Relationship: Separate models trained for row/col relationships
    """
    # Set model to train mode
    row_model.train()
    col_model.train()

    # Initialize losses to be monitored
    train_loss = 0.0
    row_loss = 0.0
    col_loss = 0.0  

    for idx, data in enumerate(train_loader):
        # Perform single step of training
        row_optimizer.zero_grad()
        col_optimizer.zero_grad()
        row_data, col_data = data
        row_data = row_data.to(DEVICE)
        col_data = col_data.to(DEVICE)

        # Gather model output and convert into a row tensor same as the ground truth
        row_logits = row_model(row_data)
        col_logits = col_model(col_data)

        # Compute individual model losses
        batch_row_loss = loss_function(row_logits, row_data.y, loss_criteria, task='sr')
        batch_col_loss = loss_function(col_logits, col_data.y, loss_criteria, task='sr')

        # Update models and optimizers
        batch_row_loss.backward()
        row_optimizer.step()

        batch_col_loss.backward()
        col_optimizer.step()

        # Aggregate losses
        row_loss += batch_row_loss.detach().item()
        col_loss += batch_col_loss.detach().item()
        train_loss += (batch_row_loss.detach().item() + batch_col_loss.detach().item())
    
    train_loss /= len(train_loader.dataset)
    row_loss /= len(train_loader.dataset)
    col_loss /= len(train_loader.dataset)

    out_dict = {
        'train_loss': train_loss,
        'row_loss': row_loss,
        'col_loss': col_loss
    }

    return out_dict


def eval_sr(row_model, col_model, val_loader, loss_criteria):
    """
    Evaluation of single relationship models
    """
    row_model.eval()
    col_model.eval()

    val_loss = 0.0
    val_row_loss = 0.0
    val_col_loss = 0.0
    val_acc = 0.0
    val_F1 = 0.0
    val_precision = 0.0
    val_recall = 0.0
    n_correct_row = 0.0
    n_total_row = 0.0
    n_correct_col = 0.0
    n_total_col = 0.0

    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            row_data, col_data = data
            row_data = row_data.to(DEVICE)
            col_data = col_data.to(DEVICE)  
            
            row_logits = row_model(row_data)
            col_logits = col_model(col_data)
            batch_row_loss = loss_function(row_logits, row_data.y, loss_criteria, task='sr')
            batch_col_loss = loss_function(col_logits, col_data.y, loss_criteria, task='sr')

            val_loss += (batch_row_loss.item() + batch_col_loss.item())
            val_row_loss += batch_row_loss.item()
            val_col_loss += batch_col_loss.item()

            # Calculate accuracy same row/col prediction
            # _, row_pred = F.softmax(row_logits.view(-1, 1), dim=0).max(1)
            # _, col_pred = F.softmax(col_logits.view(-1, 1), dim=0).max(1)
            _, row_pred = row_logits.max(1)
            _, col_pred = col_logits.max(1)
            
            row_label = row_data.y.detach().cpu().numpy()
            col_label = col_data.y.detach().cpu().numpy()

            row_pred = row_pred.detach().cpu().numpy()
            col_pred = col_pred.detach().cpu().numpy()

            n_correct_row = n_correct_row + (row_label == row_pred).sum()
            n_correct_col = n_correct_col + (col_label == col_pred).sum()
            n_total_row += row_label.shape[0]
            n_total_col += col_label.shape[0]

            # Calculate Precision, Recall and F1 Score for cell adjacency
            row_edge_index = row_data.edge_index.cpu().numpy()
            col_edge_index = col_data.edge_index.cpu().numpy()
            num_cells = row_data.pos.shape[0]
            gt_adjacency_mat = cal_adj_label(row_edge_index, col_edge_index, row_label, col_label, num_cells)
            pred_adjacency_mat = cal_adj_label(row_edge_index, col_edge_index, row_pred, col_pred, num_cells)
            
            precision = precision_score(gt_adjacency_mat.flatten(), pred_adjacency_mat.flatten(), zero_division=0)
            recall = recall_score(gt_adjacency_mat.flatten(), pred_adjacency_mat.flatten())
            if (precision + recall) != 0.0:            
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0

            val_F1 += f1
            val_precision += precision
            val_recall += recall

    val_loss /= len(val_loader.dataset)
    val_row_loss /= len(val_loader.dataset)
    val_col_loss /= len(val_loader.dataset)

    row_acc = n_correct_row / n_total_row
    col_acc = n_correct_col / n_total_col
    val_acc = 0.5 * (row_acc + col_acc)

    val_F1 /= len(val_loader.dataset)
    val_recall /= len(val_loader.dataset)
    val_precision /= len(val_loader.dataset)

    out_dict = {
        'val_loss': val_loss,
        'val_acc': val_acc,
        'row_acc': row_acc,
        'col_acc': col_acc,
        'val_f1': val_F1,
        'val_precision': val_precision,
        'val_recall': val_recall,
        'val_row_loss': val_row_loss,
        'val_col_loss': val_col_loss
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
    wandb_name = trainer_params.run + '_' + time
    wandb.init(name=wandb_name, entity='rsaha', project='table_graph_rules', config=config_dict)

    namespace_config_dict = {
                                'dataset_params': dataset_params,
                                'base_params': base_params,
                                'trainer_params': trainer_params, 
                                'img_model_params': img_model_params
                            }

    main(config=namespace_config_dict)