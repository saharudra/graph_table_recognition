import torch
import torch.nn.functional as F 
from torch_geometric.data import DataLoader

import numpy as np
from .scitsr import ScitsrDataset
from misc.args import scitsr_params, img_model_params
from models.img_models import ConvBaseGFTE

import matplotlib.pyplot as plt

dataset_params = scitsr_params()
print(dataset_params)

train_dataset = ScitsrDataset(dataset_params)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

def sample_box_features(cnnout, nodenum, pos, cell_wh, img):
    """
    cnnout:  batch_size x cnn_out_features x cnn_out_h x cnn_out_w
             Output features from image processing stem prior to sending to GCNN or MLP.
    nodenum: batch_size 
             List of length batch_size with each element giving the number of nodes (cell texts)
             in each input graph.
    pos:     num_edges x 2
             Position feature for each of the node forming an edge.
             These are original position features of the centroid of the cell text transformed
             according to F.grid_sample.

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
            cov = [[min_cellpos/16.0, min_cellpos/16.0], [min_cellpos/16.0, min_cellpos/16.0]]
            samples = np.random.multivariate_normal(cen_pos, cov, 5)
            sampling_grid = np.append(sampling_grid, samples, axis = 0)

        # Clip sampling grid to keep values between -1 and 1
        sampling_grid = np.clip(sampling_grid, -1.0, 1.0)
        # Sanity check: Scatter plot sampling_grid, imgpos
        x_imgpos_scatter = imgpos[:, 1]
        y_imgpos_scatter = imgpos[:, 0]
        x_sampling_grid = sampling_grid[:, 1]
        y_sampling_grid = sampling_grid[:, 0]
        img = img.squeeze(0).permute(1, 2, 0)
        img = img.numpy()
        plt.subplot(221)
        plt.imshow(img)
        plt.subplot(222)
        plt.scatter(x_imgpos_scatter, y_imgpos_scatter, c='red')
        plt.scatter(x_sampling_grid, y_sampling_grid, c='green')
        plt.show()
        import pdb; pdb.set_trace()

        # imgpos = imgpos.unsqueeze(0) 
        # imgpos = imgpos.unsqueeze(0)
        # cnnin = cnnout[i].unsqueeze(0)  # Single graph
        # sout = F.grid_sample(cnnin, imgpos, mode='bilinear', padding_mode='border')
        # cnt+=nodenum[i]
        # sout=sout.squeeze(0)
        # sout=sout.squeeze(1)
        # sout = sout.permute(1, 0) # num_box*feature_num
        # if i==0:
        #     out = sout
        # else:
        #     out = torch.cat((out,sout),0)
    return out

img_params = img_model_params()
print(img_params)
img_model = ConvBaseGFTE(img_params)

for idx, data in enumerate(train_loader):
    img, imgpos, nodenum, cell_wh = data.img, data.imgpos, data.nodenum, data.cell_wh
    print(img.shape)
    img_features = img_model(img)
    box_features = sample_box_features(img_features, nodenum, imgpos, cell_wh, img)
    import pdb; pdb.set_trace()
