"""
SciTSR dataset creates position features that are used as imgpos for sampling box features.

imgpos.append([(1.0 - xt[5]) * 2 - 1.0, xt[4] * 2 - 1.0])
(xt[4], xt[5]) is the centroid for each of the cell of a table.

What does plot of centroids look like? Overlaying position features on images.
Position features are in the range of (0, 1) whereas imgpos features are in the range (-1, 1).

y of imgpos features is flip of y of pos features!!!
"""
import torch
from torch_geometric.data import DataLoader

import os
import json
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ops.utils import resize_image
from misc.args import *
from dataloaders.scitsr import ScitsrDataset
from dataloaders.scitsr_transformer import ScitsrDatasetSB
from dataloaders.pubtabnet import PubTabNetDataset


"""
Plot position features for scitsr dataloader with kNN graph transformation
"""
# params = scitsr_params()
# print(params)
# train_dataset = ScitsrDataset(params)
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
# for idx, data in enumerate(train_loader):
#     pos, img, imgpos, edge_index, img_orig = data.pos, data.img, data.imgpos, data.edge_index, data.img_orig
#     print("Shape of img prior squeeze: {}".format(img.shape))
#     img = torch.squeeze(img, dim=0)
#     img_orig = torch.squeeze(img_orig, dim=0)
#     print("Shape of img after squeeze: {}".format(img.shape))
#     print("Shape of imgpos: {}".format(imgpos.shape))
#     imgpos = imgpos.numpy()
#     pos = pos.numpy()
#     img = img.permute(1, 2, 0).numpy()
#     img_orig = img_orig.permute(1, 2, 0).numpy()
#     img_resize_ar, window, scale, padding, crop = resize_image(img_orig, min_dim=1024, max_dim=1024,
#                                                             min_scale=0) 

#     # Scatter plot pos features and imgpos features
#     x_scatter = pos[:, 0] * 1024
#     # Flip along y axis
#     y_scatter = 1024 -  pos[:, 1] * 1024
#     # x_imgpos_scatter = imgpos[:, 1]
#     # y_imgpos_scatter = imgpos[:, 0] * 1.0
#     # plt.subplot(221)
#     # plt.scatter(x_imgpos_scatter, y_imgpos_scatter)
#     # plt.subplot(222)
    
#     # plot image and overlay scatter plot
#     plt.scatter(x_scatter, y_scatter, c='green')
#     plt.imshow(img)
#     plt.show()

#     h, w, c = img_orig.shape
#     x_scatter_orig = pos[:, 0] * w
#     y_scatter_orig = h - pos[:, 1] * h
#     plt.imshow(img_orig)
#     plt.scatter(x_scatter_orig, y_scatter_orig, c='red')
#     plt.show()

#     h_r, w_r, c_r = img_resize_ar.shape

#     if w > h:
#         # width > height, offset added in height or y direction
#         # scale bbox in x direction directly
#         offset = (1024 - math.floor((1024 * h) / w)) / 2
    
#     elif h > w:
#         # similarly if height > width, offset added in width or x direction
#         # scale bbox in y direction directly
#         offset = (1024 - math.floor((1024 * w) / h)) / 2
               
#     else:
#         offset = 0
    
#     if w > h:
#         y_scatter_ar = 1024 - pos[:, 1] * (1024 -  2 * offset) - offset
#         p
#         x_scatter_ar = pos[:, 0] * 1024
#     elif h > w:
#         x_scatter_ar = pos[:, 0] * (1024 - 2 * offset) - offset
#         y_scatter_ar = 1024 -  pos[:, 1] * 1024
#     else:
#         x_scatter_ar = pos[:, 0] * 1024
#         y_scatter_ar = 1024 -  pos[:, 1] * 1024

#     plt.imshow(img_resize_ar)
#     plt.scatter(x_scatter_ar, y_scatter_ar, c='blue')
#     plt.show()

#     import pdb; pdb.set_trace()

"""
Plot bounding box coordinates from original image onto resized image for PubTabNet
"""
# root = '/Users/i23271/Downloads/table/datasets/PubTabNet'
# resized_images = os.path.join(root, 'examples_trial')
# gt_path = os.path.join(root, 'examples')

# annotation_filename = gt_path + os.sep + 'PubTabNet_Examples.jsonl'

# annotations = []
# with open(annotation_filename, 'r') as jaf:
#     for line in jaf:
#         annotations.append(json.loads(line))

# for annot in annotations:
#     filename = annot['filename']
#     cell_annotation = annot['html']['cells']
#     orig_imgfn = os.path.join(gt_path, filename)
#     orig_img = cv2.cvtColor(cv2.imread(orig_imgfn), cv2.COLOR_BGR2RGB)
#     resized_imgfn = os.path.join(resized_images, filename)
#     resized_img = cv2.cvtColor(cv2.imread(resized_imgfn), cv2.COLOR_BGR2RGB)
#     h, w, c = orig_img.shape
#     h_n, w_n, c_n = resized_img.shape
#     print(h, w, c)
#     print(h_n, w_n, c_n)
#     # plt.imshow(orig_img)
#     # plt.show()

#     if w > h:
#         # width > height, offset added in height or y direction
#         # scale bbox in x direction directly
#         offset = (1024 - math.floor((1024 * h) / w)) / 2
    
#     elif h > w:
#         offset = (1024 - math.floor((1024 * w) / h)) / 2
    
#     else:
#         offset = 0
    
#     for cell in cell_annotation:
#         if 'bbox' in cell:
#             bbox = cell['bbox']
#             x0, y0, x1, y1 = bbox

#             if w > h:
#                 x0 = int((x0 / w) * w_n)
#                 x1 = int((x1 / w) * w_n)
#                 y0 = int( offset + ((y0 / h) * math.floor((1024 * h) / w)) )
#                 y1 = int( offset + ((y1 / h) * math.floor((1024 * h) / w)) )

#             elif h > w:
#                 x0 = int( offset + ((x0 / w) * math.floor((1024 * w) / h)) )
#                 x1 = int(offset + ((x1 / w ) * math.floor((1024 * w) / h)) )
#                 y0 = int((y0 / h) * h_n)
#                 y1 = int((y1 / h) * h_n)
            
#             else:
#                 x0 = int((x0 / w) * w_n)
#                 x1 = int((x1 / w) * w_n)
#                 y0 = int((y0 / h) * h_n)
#                 y1 = int((y1 / h) * h_n)

#             cv2.rectangle(resized_img, (x0, y0), (x1, y1), (0, 0, 255), 2)
    
    
#     plt.imshow(resized_img)
#     plt.show()
#     # import pdb; pdb.set_trace()


"""
Plot position features for scitsr dataloader for the transformer based model
"""

params = scitsr_params()
print(params)
train_dataset = ScitsrDatasetSB(params)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

for idx, data in enumerate(train_loader):
    pos, imgpos, img = data.pos, data.imgpos, data.img
    rescale_params = data.rescale
    rescale_params = rescale_params.numpy()
    img = torch.squeeze(img, dim=0)
    # h_o, w_o, h_n, w_n = rescale_params
    imgpos = imgpos.numpy()
    pos = pos.numpy()
    img = img.permute(1, 2, 0).numpy()
    # chk_limit = chk_limit.numpy()
    
    x_scatter = pos[:, 0] * 1024
    y_scatter = pos[:, 1] * 1024

    # plot image and overlay scatter plot
    fig, ax = plt.subplots()
    ax.scatter(x_scatter, y_scatter, c='green')
    ax.imshow(img)
    # rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
    # ax.add_patch(rect)
    plt.show()
    import pdb; pdb.set_trace()


"""
Plot position features for pubtabnet dataloader for the transformer based model
"""
# params = pubtabnet_parms()
# print(params)
# val_dataset = PubTabNetDataset(params, partition='val')
# val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)

# for idx, data in enumerate(val_loader):
#     pos, imgpos, img, cl = data.pos, data.imgpos, data.img, data.cl
#     img = img[0]
#     c, w, h = img.shape
#     imgpos = imgpos.numpy()
#     pos = pos.numpy()
#     cl = cl.numpy()
#     x_min, x_max, y_min, y_max, width, height, avhei = cl
#     print(cl)
#     img = img.permute(1, 2, 0).numpy() 
#     print(img.dtype)
#     x_scatter = pos[:, 0] * w
#     y_scatter = pos[:, 1] * h

#     fig, ax = plt.subplots()
#     ax.scatter(x_scatter, y_scatter, c='green')
#     ax.imshow(img)
#     rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
#     ax.add_patch(rect)
#     plt.show()
#     import pdb; pdb.set_trace()

