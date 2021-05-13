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

from models.graph_rules import GraphRulesSingleRelationship, GraphRulesMultiLabel, GraphRulesMultiTask
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
    
    # Define loss criteria
    if trainer_params.loss_criteria == 'nll':
        loss_criteria = nn.NLLLoss()
    elif trainer_params.loss_criteria == 'bce_logits':
        weight_tensor = torch.Tensor([trainer_params.class_weight]).to(DEVICE)
        loss_criteria = nn. BCEWithLogitsLoss(pos_weight=weight_tensor, reduction='sum')

    if dataset_params.gr_single_relationship:
        eval_out_dict = eval_sr(row_model, col_model, eval_loader, loss_criteria)
        print(eval_out_dict)


def eval_sr(row_model, col_model, eval_loader, loss_criteria):
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
        for idx, data in enumerate(eval_loader):
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

    val_loss /= len(eval_loader.dataset)
    val_row_loss /= len(eval_loader.dataset)
    val_col_loss /= len(eval_loader.dataset)

    row_acc = n_correct_row / n_total_row
    col_acc = n_correct_col / n_total_col
    val_acc = 0.5 * (row_acc + col_acc)

    val_F1 /= len(eval_loader.dataset)
    val_recall /= len(eval_loader.dataset)
    val_precision /= len(eval_loader.dataset)

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