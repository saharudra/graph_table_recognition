import argparse

def scitsr_params():
    parser = argparse.ArgumentParser(description="Arguments for prepairing SciTSR table recognition task dataset")

    # Base arguments
    parser.add_argument('--run', type=str, default='version_1_a',
                        help='model version to be run')
    parser.add_argument('--exp', type=str, default='table_recognition',
                        help='task to be run')
    parser.add_argument('--seed', type=int, default=1234, 
                        help='seed value for reproducibility')
    parser.add_argument('--device', type=str, default='cpu',
                        help='device to run the code on cuda | cpu')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='/Users/i23271/Downloads/table/datasets/SciTSR',
                        help='data directory')

    # Data processing arguments
    parser.add_argument('--text_encode_len', type=int, default=15,
                        help='max length for encoding text')
    parser.add_argument('--img_size', type=int, default=256,
                        help='size of the image taken in by the model')
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='size of the kernel for dilation or erosion')
    parser.add_argument('--graph_k', type=int, default=6,
                        help='K-value for KNN graph transform')
    parser.add_argument('--dilate', type=bool, default=False,
                        help='whether to dilate images or not')
    parser.add_argument('--erode', type=bool, default=False,
                        help='whether to erode images or not, to thicken lines and text')
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
    parser.add_argument('--sp', nargs="+", default=[2, 2, 2])

    opt = parser.parse_args()

    return opt
    