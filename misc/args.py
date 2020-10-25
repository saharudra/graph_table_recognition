import argparse

def scitsr_params():
    parser = argparse.ArgumentParser(description="Arguments for prepairing SciTSR table recognition task dataset")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='/datatop_1/rudra/table_recognition/datasets/SciTSR',
                        help='data directory')

    # Data processing arguments
    parser.add_argument('--alphabet', type=str, default="0123456789abcdefghijklmnopqrstuvwxyz,.*# ",
                        help='characters that are being encoded')
    parser.add_argument('--text_encode_len', type=int, default=15,
                        help='max length for encoding text')
    parser.add_argument('--img_size', type=int, default=256,
                        help='size of the image taken in by the model')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='size of the kernel for dilation or erosion')
    parser.add_argument('--device', type=str, default='cpu',
                        help='device to run graph transform on, currently cpu')
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
    parser.add_argument('--device', type=str, default='cpu',
                        help='device to run the code on cuda | cpu')
    
    # Base img model arguments
    parser.add_argument('--inc', type=int, default=3,
                        help='number of input channels of the image')

    # GFTE img model arguments
    parser.add_argument('--ks', nargs='+', default=[3, 3, 3],
                        help='kernel size for convolution')
    parser.add_argument('--nif', nargs="+", default=[64, 64, 64],
                        help='number of features for each conv operation')
    parser.add_argument('--ss', nargs="+", default=[1, 1, 1],
                        help='stride for convolution')
    parser.add_argument('--ps', nargs="+", default=[1, 1, 1, 1],
                        help='padding for convolution')
    parser.add_argument('--sp', nargs="+", default=[2, 2, 2],
                        help='stride for pooling')

    opt = parser.parse_args()

    return opt


def base_params():
    parser = argparse.ArgumentParser(description="Arguments for table structure recognition base")

    # Global params
    parser.add_argument('--num_classes', type=int, default=2,
                        help='both row and col classification as binary classification')
    parser.add_argument('--version', type=str, default='v_1_b',
                        help='which version of the current model config. v_1_a | v_1_b | v_1_c')

    # Position feature params 
    parser.add_argument('--num_hidden_features', type=int, default=64, 
                        help='number of hidden features for the entire processing')
    parser.add_argument('--num_node_features', type=int, default=8,
                        help='number of input features of a node')

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
    
    opt = parser.parse_args()

    return opt


def train_params():
    parser = argparse.ArgumentParser(description="Arguments for trainer scripts")

    # Base arguments
    parser.add_argument('--exp', type=str, default='table_structure_recognition',
                        help='task to be run, defines save directory root.')
    parser.add_argument('--run', type=str, default='version_1_row_only',
                        help='model version to be run')
    parser.add_argument('--seed', type=int, default=1234, 
                        help='seed value for reproducibility')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to run the code on cuda | cpu')

    # Training type arguments
    parser.add_argument('--row_only', type=bool, default=False,
                        help='trains a row only model')
    parser.add_argument('--col_only', type=bool, default=False,
                        help='trains a col only model')
    parser.add_argument('--multi_task', type=bool, default=True,
                        help='trains a multi-task model')

    # Dataloader arguments
    parser.add_argument('--workers', type=int, default=0,
                        help='number of dataloading workers, not an option in torch_geometric')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size')

    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=100, 
                        help='number of epochs to train for')

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
    parser.add_argument('--lr_cooldonw', type=int, default=2,
                        help='number of epochs to wait before resuming normal operations after lr reduction')
    parser.add_argument('--min_lr', type=float, default=1e-5,
                        help='minimum lr for learning rate scheduling')

    parser.add_argument('--optimizer', type=str, default='adam',
                        help='optimizer to use: adam | adadelta | rmsprop')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='beta2 for adam optimizer')

    parser.add_argument('--logging_interval', type=int, default=20,
                        help='interval at which to log information')
    parser.add_argument('--val_interval', type=int, default=10,
                        help='interval at which validation will be performed and logged')
    parser.add_argument('--save_interval', type=int, default=1,
                        help='interval at which model check will be done to save best')

    opt = parser.parse_args()

    return opt
    
    

    
    

    