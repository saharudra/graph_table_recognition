import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F 
from torch_geometric.data import DataLoader

import numpy as np
import numpy_indexed as npi
from .scitsr import ScitsrDataset
from misc.args import scitsr_params, img_model_params
from models.img_models import ConvBaseGFTE

import matplotlib.pyplot as plt

dataset_params = scitsr_params()
print(dataset_params)

train_dataset = ScitsrDataset(dataset_params)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

def sample_box_features(cnnout, nodenum, pos, cell_wh, img, num_samples=5, div=16.0):
    """
    TODO: Make the feature efficient. Start by removing internal loops.
    cnnout:  batch_size x cnn_out_features x cnn_out_h x cnn_out_w
             Output features from image processing stem prior to sending to GCNN or MLP.

    nodenum: batch_size 
             List of length batch_size with each element giving the number of nodes (cell texts)
             in each input graph.

    pos:     num_edges x 2
             Position feature for each of the node forming an edge.
             These are original position features of the centroid of the cell text transformed
             according to F.grid_sample.

    cell_wh: num_edges x 2
             Height and width information for each of the node i.e cell text.
             Using the min of this to create the isotropic gaussian to sample from each node.

    img:     batch_size x 3 x img_h x img_w x

    num_samples: Number of samples to take from each node.

    div:      Division for covariance matrix, defines kurtosis.

    out:     num_edges x sampled_img_features
             If sampling only from the centroid of the cell text, sampled_img_features == 64.
             Sampling from multiple positions and concatenating to a given node feature.
    """
    cnt = 0
    # Sampling from one graph at a time
    for i in range(nodenum.size()[0]):
        imgpos=pos[cnt:cnt+nodenum[i], :]
        cellpos = cell_wh[cnt:cnt+nodenum[i], :]
        
        # Create sampling grid using gaussian sampling around cell text center 
        # Use cell text height and width to define the covariance matrix for sampling
        sampling_grid = np.empty((0, 2))
        imgpos = imgpos.numpy()
        cellpos = cellpos.numpy()
        for idx, cen_pos in enumerate(imgpos):
            min_cellpos = min(cellpos[idx])
            # Sample from an isotropic gaussian with mean as cell text centroid 
            cov = [[min_cellpos/div, 0], [0, min_cellpos/div]]
            samples = np.random.multivariate_normal(cen_pos, cov, num_samples)
            sampling_grid = np.append(sampling_grid, samples, axis = 0)

        # Clip sampling grid to keep values between -1 and 1
        sampling_grid = np.clip(sampling_grid, -1.0, 1.0)

        # Sanity check: Scatter plot sampling_grid, imgpos
        # x_imgpos_scatter = imgpos[:, 1]
        # y_imgpos_scatter = imgpos[:, 0]
        # x_sampling_grid = sampling_grid[:, 1]
        # y_sampling_grid = sampling_grid[:, 0]
        # img = img[0].squeeze(0).permute(1, 2, 0)
        # img = img.numpy()
        # plt.subplot(221)
        # plt.imshow(img)
        # plt.subplot(222)
        # plt.scatter(x_imgpos_scatter, y_imgpos_scatter, c='red')
        # plt.scatter(x_sampling_grid, y_sampling_grid, c='green')
        # plt.show()
        # import pdb; pdb.set_trace()
        
        # Conver back to torch tensor
        imgpos = torch.FloatTensor(imgpos)
        sampling_grid = torch.FloatTensor(sampling_grid)
        sampling_grid = sampling_grid.unsqueeze(0) 
        sampling_grid = sampling_grid.unsqueeze(0)
        cnnin = cnnout[i].unsqueeze(0)  # Single graph
        sout = F.grid_sample(cnnin, sampling_grid, mode='bilinear', padding_mode='border')
        cnt+=nodenum[i]
        sout=sout.squeeze(0)
        sout=sout.squeeze(1)
        sout = sout.permute(1, 0) # (num_box * num_samples) x feature_num
        # idx = torch.FloatTensor(np.repeat(np.arange(1, nodenum[i].item() + 1), num_samples)).view(nodenum[i].item() * num_samples, -1)
        # sout = torch.cat([idx, sout], axis=1)
        size = sout.size()
        sampling_out = torch.empty((0, size[-1]))
        sample_lst = []
        for it, curr_out in enumerate(sout):
            curr_out = curr_out.reshape(1, size[-1])
            if it == 0:
                sampling_out = torch.cat([sampling_out, curr_out], axis=0)
            elif it % num_samples != 0:
                sampling_out = torch.cat([sampling_out, curr_out], axis=1)
            else:
                sample_lst.append(sampling_out)
                sampling_out = torch.empty((0, size[-1]))
                sampling_out = torch.cat([sampling_out, curr_out], axis=0)
        sample_lst.append(sampling_out)
        sample_out = torch.cat(sample_lst, axis=0)
        # sout = npi.group_by(sout[:, 0]).split(sout[:, 1:]).reshape(nodenum[i].item(), -1)
        # sout = torch.FloatTensor(sout)
        if i==0:
            out = sample_out
        else:
            out = torch.cat((out,sample_out),0)
    return out

img_params = img_model_params()
print(img_params)
img_model = ConvBaseGFTE(img_params)

for idx, data in enumerate(train_loader):
    img, imgpos, nodenum, cell_wh = data.img, data.imgpos, data.nodenum, data.cell_wh
    img_features = img_model(img)
    box_features = sample_box_features(img_features, nodenum, imgpos, cell_wh, img, num_samples=5, div=32.0)
    import pdb; pdb.set_trace()
