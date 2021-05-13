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
# import wandb
from datetime import datetime
from tqdm import trange
# from sklearn.metrics import precision_score, recall_score

from models.graph_rules import GraphRulesMultiClass
from dataloaders.graph_rules_multi_class import GraphRulesMultiClassLoader
from misc.args import scitsr_params, base_params, trainer_params, img_model_params
from ops.misc import weights_init, mkdir_p

def main(config):
    dataset_params = config['dataset_params']
    base_params = config['base_params']
    trainer_params = config['trainer_params']
    img_model_params = config['img_model_params']

    # Define dataloader
    train_dataset = GraphRulesMultiClassLoader(dataset_params, partition='train')
    train_loader = DataLoader(train_dataset, batch_size=trainer_params.batch_size, shuffle=True,
                              num_workers=trainer_params.workers, pin_memory=trainer_params.pin_memory)
    
    val_dataset = GraphRulesMultiClassLoader(dataset_params, partition='test')
    val_loader =  DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Define model and initialize weights
    if dataset_params.gr_multi_class:
        model = GraphRulesMultiClass(base_params)
        model.apply(weights_init)
        model.to(DEVICE)
    else:
        print('Incorrect Argument Set. Check misc/args.py')
    
    if trainer_params.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=trainer_params.lr,
                                betas=(trainer_params.beta1, trainer_params.beta2))

    if trainer_params.schedule_lr:
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=trainer_params.lr_patience,
                                          factor=trainer_params.lr_reduce_factor, verbose=True, 
                                            mode=trainer_params.lr_schedule_mode,
                                            cooldown=trainer_params.lr_cooldown, min_lr=trainer_params.min_lr)

    # Define loss criteria
    loss_criteria = nn.CrossEntropyLoss()
    
    # Watch model 
    # wandb.watch(model)

    # Model saving params
    best_accuracy = 0.0
    best_epoch = -1

    if trainer_params.overfit_one_batch:
        print("$$$ OVERFITTING ON ONE BATCH $$$")
        data = next(iter(train_loader))
        with trange(trainer_params.num_epochs) as t:
            for epoch in t:
                train_out_dict = overfit_one_batch(model, optimizer, data, loss_criteria)

                # wandb.log(train_out_dict)
                t.set_postfix(train_out_dict)
                t.update()
    else:
        print('LENGTH OF TRAINING DATASET: {}, VALIDATION DATASET: {}'.format(len(train_loader.dataset), len(val_loader.dataset)))
        print('$$$ STARTING TRAINING LOOP $$$')

        with trange(trainer_params.num_epochs) as t:
            for epoch in t:
                # Train a single pass of the model
                train_out_dict = train(model, optimizer, train_loader, loss_criteria)
                # wandb.log(train_out_dict)

                # Perform evaluation at intervals
                if epoch % trainer_params.val_interval == 0:
                    eval_out_dict = eval(model, val_loader, loss_criteria)
                    # wandb.log(eval_out_dict)
                
                # Schedule learning rate
                if trainer_params.schedule_lr:
                    lr_scheduler.schedule_lr(train_out_dict['train_loss'])

                # Save models based on accuracy/F1 score
                if epoch % trainer_params.val_interval == 0:
                    if eval_out_dict['val_acc'] > best_accuracy:
                        
                        model_path = log_path + os.sep + "net_{}_best_so_far.pth".format(epoch)
                        torch.save(model.state_dict(), model_path)

                        best_accuracy = eval_out_dict['val_acc']
                        best_epoch = epoch
                
                t_postfix_dict = {
                    'train_loss': train_out_dict['train_loss'],
                    'val_acc': eval_out_dict['val_acc'],
                }
                t.set_postfix(t_postfix_dict)
                t.update()
        
        print("*** TRAINING IS COMPLETE ***")
        print("Best validation accuracy: {}, in epoch: {}".format(best_accuracy, best_epoch))


def overfit_one_batch(model, optimizer, data, loss_criteria):
    # Get data
    data = data.to(DEVICE)
    # Flush old gradients
    optimizer.zero_grad()

    # Compute model losses
    logits = model(data)
    batch_loss = loss_function(logits, data.gt, loss_criteria)

    # Update models and optimizers
    batch_loss.backward()
    optimizer.step()
    
    # Aggregate losses
    loss = batch_loss.detach().item() / len(data.gt)

    # Calculate accuracy
    pred = torch.argmax(logits)
    





def train():
    pass

def eval():
    pass

def loss_function(logits, gt, loss_criteria):
    loss = loss_criteria(logits, gt)
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