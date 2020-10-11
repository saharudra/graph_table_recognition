"""
SciTSR dataset creates position features that are used as imgpos for sampling box features.

>>> imgpos.append([(1.0 - xt[5]) * 2 - 1.0, xt[4] * 2 - 1.0])
(xt[4], xt[5]) is the centroid for each of the cell of a table.

What does plot of centroids look like? Overlaying position features on images.
Position features are in the range of (0, 1) whereas imgpos features are in the range (-1, 1).

y of imgpos features is flip of y of pos features!!!
"""
import torch
from torch_geometric.data import DataLoader

import matplotlib.pyplot as plt

from misc.args import *
from dataloaders.scitsr import ScitsrDataset

params = scitsr_params()
print(params)

train_dataset = ScitsrDataset(params)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

for idx, data in enumerate(train_loader):
    pos, img, imgpos = data.pos, data.img, data.imgpos
    print("Shape of img prior squeeze: {}".format(img.shape))
    img = torch.squeeze(img, dim=0)
    print("Shape of img after squeeze: {}".format(img.shape))
    print("Shape of imgpos: {}".format(imgpos.shape))
    imgpos = imgpos.numpy()
    pos = pos.numpy()
    img = img.numpy()
    # Scatter plot pos features and imgpos features
    x_scatter = pos[:, 0] 
    y_scatter = pos[:, 1] 
    x_imgpos_scatter = imgpos[:, 1]
    y_imgpos_scatter = imgpos[:, 0] * 1.0
    plt.subplot(221)
    plt.scatter(x_imgpos_scatter, y_imgpos_scatter)
    plt.subplot(222)
    plt.scatter(x_scatter, y_scatter)
    # plt.imshow(img)
    plt.show()
    import pdb; pdb.set_trace()



