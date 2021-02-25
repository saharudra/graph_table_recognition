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
    # get upper triangular matrix to create 
    # indices from the tensor to be paired
    row = len(tensor)
    r, c = np.triu_indices(row, 1)
    out = torch.cat((tensor[r], tensor[c]), dim=1)
    return out


if __name__ == '__main__':
    arr = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
    out = pairwise_combinations(arr).numpy()
    print(out)