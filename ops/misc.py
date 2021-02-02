import torch
import os 
import errno
import numpy as np

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

# custom weight initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def pairwise_combinations(tensor):
    # Numpy version for an array
    row, col = tensor.shape
    print(row, col) 
    r, c = torch.triu_indices(row, col, 0)
    out = torch.hstack((tensor[r], tensor[c]))
    return out


if __name__ == '__main__':
    arr = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]])
    out = pairwise_combinations(arr)
    print(out)