import torch
import torch.nn as nn
import torch_geometric.nn as tgnn

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
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
    elif isinstance(m, tgnn.GCNConv):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def pairwise_combinations(tensor, batched=False):
    if not batched:
        # get upper triangular matrix to create 
        # indices from the tensor to be paired
        row = len(tensor)
        r, c = np.triu_indices(row, 1)
        out = torch.cat((tensor[r], tensor[c]), dim=1)
        return out
    else:
        # Perform the same for each of the rows for the tensor
        # TODO: Vectorize this!
        out_tensor = []
        for sample in tensor:
            row = len(sample)
            r, c = np.triu_indices(row, 1)
            out = torch.cat((sample[r], sample[c]), dim=1)
            out_tensor.append(out)
        out_tensor = torch.cat([sample.unsqueeze(0) for sample in out_tensor], dim=0)
        return out_tensor


if __name__ == '__main__':
    arr = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [19, 20, 21]], 
                       [[10, 11, 12], [13, 14, 15], [16, 17, 18], [22, 23, 24]]])
    # arr = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print("Shape of input {}".format(arr.shape))
    out = pairwise_combinations(arr, batched=True).numpy()
    print("Shape of final output {}".format(out.shape))