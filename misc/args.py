import argparse

def scitsr_params():
    parser = argparse.ArgumentParser(description="arguments for training SciTSR table recognition task")

    # Base arguments
    parser.add_argument('--run', type=str, default='version_1_a',
                        help='model version to be run')
    parser.add_argument('--exp', type=str, default='table_recognition',
                        help='task to be run')
    parser.add_argument('--seed', type=int, default=1234, 
                        help='seed value for reproducibility')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device to run the code on cuda | cpu')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='/datatop_1/rudra/table_recognition/datasets/SciTSR',
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
    