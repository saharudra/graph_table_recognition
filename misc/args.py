import argparse

def pubtabnet_parms():
    parser = argparse.ArgumentParser(description="Arguments for prepairing PubTabNet table structure recognition dataset")

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='/Users/i23271/Downloads/table/datasets/pubtabnet',
                        help='data directory')
    parser.add_argument('--json_file', type=str, default='PubTabNet_2.0.0.jsonl',
                        help='Annotation file for all splits')
    
    parser.add_argument('--new_imglist', type=bool, default=False,
                        help='whether to create a new imglist or use existing one if it exists')
    parser.add_argument('--img_size', type=int, default=1024,
                        help='size of the image taken in by the model')
    parser.add_argument('--img_resize_mode', type=str, default='square',
                        help='ways to resize the image. none | square | pad64 | crop')
    parser.add_argument('--img_scale', type=int, default=0,
                        help='ensure that the image is scaled up by at least this percent even if \
                              min_dim doesnot require it.')
    parser.add_argument('--augment_chunk', type=bool, default=False, 
                        help='whether to jitter position of cell text bounding box')
                        
    opt = parser.parse_args()

    return opt
    

def scitsr_params():
    parser = argparse.ArgumentParser(description="Arguments for prepairing SciTSR table recognition task dataset")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='/Users/i23271/Downloads/table/datasets/SciTSR',
                        help='data directory')

    # Data processing arguments
    parser.add_argument('--alphabet', type=str, default="0123456789abcdefghijklmnopqrstuvwxyz,.*# ",
                        help='characters that are being encoded')
    parser.add_argument('--text_encode_len', type=int, default=15,
                        help='max length for encoding text')
    parser.add_argument('--img_size', type=int, default=1024,
                        help='size of the image taken in by the model')
    parser.add_argument('--img_resize_mode', type=str, default='square',
                        help='ways to resize the image. none | square | pad64 | crop')
    parser.add_argument('--img_scale', type=int, default=0,
                        help='ensure that the image is scaled up by at least this percent even if \
                              min_dim doesnot require it.')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='size of the kernel for dilation or erosion')
    parser.add_argument('--graph_k', type=int, default=6,
                        help='K-value for KNN graph transform')
    parser.add_argument('--dilate', type=bool, default=False,
                        help='whether to dilate images or not')
    parser.add_argument('--erode', type=bool, default=False,
                        help='whether to erode images or not, to thicken lines and text')
    parser.add_argument('--labeling_sanity', type=bool, default=True,
                        help='sanity checking labeling when cell text is split.')
    parser.add_argument('--augment_chunk', type=bool, default=False, 
                        help='whether to jitter position of cell text bounding box')
    parser.add_argument('--new_imglist', type=bool, default=False,
                        help='whether to create a new imglist or use existing one if it exists')

    parser.add_argument('--padding', type=bool, default=True,
                        help='whether to pad tokens or not')
    parser.add_argument('--padding_value', type=float, default=-1.0,
                        help='padding value for creating equal sequence length (number of tokens aka cells) for each table')
    parser.add_argument('--padding_length', type=int, default=25,
                        help='padding to create equal sequence length')

    # Params for imposing rules to create sets
    parser.add_argument('--rules_constraint', type=str, default='naive_gaussian',
                        help='the rule being applied to create sets')
    
    # Params for naive_gaussian rule
    parser.add_argument('--ng_mean_pos', type=str, default='centroid',
                        help='where to put the mean location for 1-D Gaussians, centroid | tlbr')

    # Type of Graph generation rules params
    parser.add_argument('--gr_single_relationship', type=bool, default=False,
                        help='row only/col only models')
    parser.add_argument('--gr_multi_class', type=bool, default=True,
                        help='graph combines both row and col adjacency rule matrices\
                              edges have both row and column labels')
    parser.add_argument('--gr_multi_task', type=bool, default=False,
                        help='graph combines both row and col adjacency rule matrices\
                              edges are transformed using a common stem and heads added\
                              for row/col classification tasks')

    opt = parser.parse_args()

    return opt


def img_model_params():
    parser = argparse.ArgumentParser(description="Arguments for image processing modules")

    # Base arguments
    parser.add_argument('--run', type=str, default='version_1_a',
                        help='model version to be run')
    parser.add_argument('--exp', type=str, default='table_recognition',
                        help='task to be run')
    parser.add_argument('--seed', type=int, default=1234, 
                        help='seed value for reproducibility')
    
    # Base img model arguments
    parser.add_argument('--inc', type=int, default=3,
                        help='number of input channels of the image')

    # GFTE img model arguments
    parser.add_argument('--ks', nargs='+', default=[3, 3, 3],
                        help='kernel size for convolution')
    parser.add_argument('--nif', nargs="+", default=[64, 64, 128],
                        help='number of features for each conv operation')
    parser.add_argument('--ss', nargs="+", default=[1, 1, 1],
                        help='stride for convolution')
    parser.add_argument('--ps', nargs="+", default=[1, 1, 1, 1],
                        help='padding for convolution')
    parser.add_argument('--sp', nargs="+", default=[2, 2, 2],
                        help='stride for pooling')

    # ResNet img model arguments
    parser.add_argument('--resnet', type=bool, default=True,
                        help='whether to use resnet model for image processing')
    parser.add_argument('--resnet_model', type=str, default='resnet18',
                        help='resnet model to be used: resnet18, resnet50, \
                            resnext50_32x4d, wide_resnet50_2')
    parser.add_argument('--resnet_out_layer', type=int, default=3,
                        help='resnet layer to get image features from')
    parser.add_argument('--resnet_pretrained', type=bool, default=False,
                        help='resent model pre-trained with ImageNet features')

    opt = parser.parse_args()

    return opt


def base_params():
    parser = argparse.ArgumentParser(description="Arguments for table structure recognition base")

    # Global params
    parser.add_argument('--num_classes', type=int, default=2,
                        help='both row and col classification as binary classification')

    # Position feature params 
    parser.add_argument('--num_hidden_features', type=int, default=128, 
                        help='number of hidden features for the entire processing')
    parser.add_argument('--num_node_features', type=int, default=8,
                        help='number of input features of a node in graph models')
    parser.add_argument('--num_pos_features', type=int, default=2,
                        help='number of pos features for input to transformer model')

    # Text feature params
    parser.add_argument('--vocab_size', type=int, default=41,
                        help='vocabulary size based on number of characters being compared')
    parser.add_argument('--num_text_features', type=int, default=64,
                        help='number of text features for input')
    parser.add_argument('--bidirectional', type=bool, default=False,
                        help='whether to consider a bidirectional rnn')

    # Image feature params
    parser.add_argument('--num_samples', type=int, default=1,
                        help='number of points to sample image features from for each cell text')
    parser.add_argument('--div', type=float, default=16.0,
                        help='defining kurtosis of each of the isotropic gaussian distribution')

    # Transformer model params
    parser.add_argument('--transformer', type=bool, default=True,
                        help='whether to use the transformer model')
    parser.add_argument('--num_encoder_layers', type=int, default=2,
                        help='number of encoder layers in the transformer model')
    parser.add_argument('--transformer_norm', type=str, default=None,
                        help='norm for the transformer encoder layer')
    parser.add_argument('--num_attn_heads', type=int, default=2,
                        help='number of attention heads in each of the encoder layers')
    
    # Set transformer model params
    parser.add_argument('--set_transformer', type=bool, default=False,
                        help='whether to use set transformer model')
    parser.add_argument('--num_set_inds', type=int, default=32,
                        help='number of induced indices for set transformer')
    parser.add_argument('--num_set_attn_heads', type=int, default=4, 
                        help='number of attention heads for set transformer model')
    parser.add_argument('--num_set_hidden_features', type=int, default=128,
                        help='number of hidden features for the set transformer model')

    opt = parser.parse_args()

    return opt


def trainer_params():
    parser = argparse.ArgumentParser(description="Arguments for trainer scripts")

    # Base arguments
    parser.add_argument('--exp', type=str, default='graph_rules_results',
                        help='task to be run, defines save directory root.')
    parser.add_argument('--run', type=str, default='multi_class',
                        help='model version to be run')
    parser.add_argument('--overfit_one_batch', type=bool, default=True,
                        help='whether overfitting the model under consideration on a single batch')
    parser.add_argument('--seed', type=int, default=1234, 
                        help='seed value for reproducibility')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to run the code on cuda | cpu')

    # Training type arguments for GFTE variants
    # parser.add_argument('--maj_ver', type=str, default='2',
    #                     help='major model version to train: 1 | 2 ...')
    # parser.add_argument('--min_ver', type=str, default='c',
    #                     help='minor model version to train 2."a" ...: a | b | c | ...')
    parser.add_argument('--row_only', type=bool, default=False,
                        help='trains above model with row only loss and reports row acc.')
    parser.add_argument('--col_only', type=bool, default=True,
                        help='trains above model with col only loss and reports col acc.')
    parser.add_argument('--multi_task', type=bool, default=False,
                        help='trains above model with multi-task loss and reports row/col acc.')

    # Dataloader arguments
    parser.add_argument('--dataset', type=str, default='scitsr',
                        help='dataset to be used for training and validation scitsr | pubtabnet')
    parser.add_argument('--eval_dataset', type=str, default='icdar2013',
                        help='dataset to be used for evaluating and benchmarking models icdar2013 | icdar2019')
    parser.add_argument('--workers', type=int, default=0,
                        help='number of dataloading workers, not an option in torch_geometric')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='input batch size')
    parser.add_argument('--pin_memory', type=bool, default=False, 
                        help='https://stackoverflow.com/questions/55563376/pytorch-how-does-pin-memory-work-in-dataloader')
    parser.add_argument('--batched', type=bool, default=True,
                        help='whether models are processing samples in a batched manner or one at a time.')

    # Training arguments
    # Lenth of training arguments
    parser.add_argument('--num_epochs', type=int, default=500, 
                        help='number of epochs to train for')
    parser.add_argument('--early_stopping', type=bool, default=False,
                        help='whether to perform early_stopping')

    # Learning rate arguments
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='initial learning rate for training')
    parser.add_argument('--schedule_lr', type=bool, default=True,
                        help='whether to perform learning rate scheduling')
    parser.add_argument('--lr_patience', type=int, default=5,
                        help='number of epochs of no improvement after which lr will be reduced')
    parser.add_argument('--lr_reduce_factor', type=float, default=0.5,
                        help='factor by which to reduce the learning rate')
    parser.add_argument('--lr_schedule_mode', type=str, default='min',
                        help='monitoring train loss, if monitoring train accuracy convert to max')
    parser.add_argument('--lr_cooldown', type=int, default=2,
                        help='number of epochs to wait before resuming normal operations after lr reduction')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='minimum lr for learning rate scheduling')

    # Optimizer arguments
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer to use: adam | adadelta | rmsprop')
    parser.add_argument('--optimizer_accu_steps', type=int, default=64,
                        help='number of examples to accumulate gradients for')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='beta2 for adam optimizer')

    # Loss arguments
    parser.add_argument('--loss_criteria', type=str, default='nll',
                        help='loss criteria to be used; bce_logits | nll | multi-label')
    parser.add_argument('--class_weight', type=float, default=100.0,
                        help='weigtage applied to the positive class as more 1s than 0s')

    # Logging arguments
    parser.add_argument('--val_interval', type=int, default=5,
                        help='interval at which validation will be performed and logged')
    parser.add_argument('--infer_interval', type=int, default=50,
                        help='interval at which granular accuracy calculation and evaluation metrics (F1) calculation' + \
                        'will be performed')
    parser.add_argument('--save_interval', type=int, default=1,
                        help='interval at which model check will be done to save best')

    opt = parser.parse_args()

    return opt


def evaluation_params():
    parser = argparse.ArgumentParser(description="Arguments for evaluation scripts")

    # saving arguments
    parser.add_argument('--exp', type=str, default='icdar_2013_comp',
                        help='which evaluation experiment is being run')

    # model specific arguments
    parser.add_argument('--pretrained_root', type=str, 
                        default='/data/rudra/table_structure_recognition/graph_table_recognition/graph_rules_results/precision_recall_checks_2021_04_20_18_30/checkpoints',
                        help='path to the root of pretrained_model')
    parser.add_argument('--best_epoch', type=int, default=900,
                        help='best epoch model to take')

    # dataset specific arguments
    parser.add_argument('--eval_dataset', type=str, default='icdar2013',
                        help='dataset used for evaluation. icdar2013 | icdar2019  | pubtabnet')
    parser.add_argument('--data_dir', type=str, default='/data/rudra/table_structure_recognition/datasets/icdar/2013_eu  ',
                          help='eval_data_dir')
    

    opt = parser.parse_args()

    return opt
    

    
    

    