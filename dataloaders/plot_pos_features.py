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
import math
import matplotlib.pyplot as plt

from misc.args import *
from dataloaders.scitsr_transformer import ScitsrDatasetSB

params = scitsr_params()
print(params)

train_dataset = ScitsrDatasetSB(params)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

"""
Plot position features for scitsr dataloader with kNN graph transformation
"""
# for idx, data in enumerate(train_loader):
#     pos, img, imgpos, edge_index = data.pos, data.img, data.imgpos, data.edge_index
#     print("Shape of img prior squeeze: {}".format(img.shape))
#     img = torch.squeeze(img, dim=0)
#     print("Shape of img after squeeze: {}".format(img.shape))
#     print("Shape of imgpos: {}".format(imgpos.shape))
#     imgpos = imgpos.numpy()
#     pos = pos.numpy()
#     img = img.permute(1, 2, 0).numpy()
#     # Scatter plot pos features and imgpos features
#     x_scatter = pos[:, 0] * 256
#     # Flip along y axis
#     y_scatter = 256 -  pos[:, 1] * 256
#     # x_imgpos_scatter = imgpos[:, 1]
#     # y_imgpos_scatter = imgpos[:, 0] * 1.0
#     # plt.subplot(221)
#     # plt.scatter(x_imgpos_scatter, y_imgpos_scatter)
#     # plt.subplot(222)
    
#     # plot image and overlay scatter plot
#     plt.scatter(x_scatter, y_scatter, c='green')
#     plt.imshow(img)
    
#     # plot edges and save the images
#     edge_index = edge_index.numpy()
#     edge_count = 0
#     for edge in range(edge_index.shape[1]):
#         edge_nodes = edge_index[:, edge]
#         start_node_x = pos[edge_nodes[1], 0] * 256
#         start_node_y = 256 - pos[edge_nodes[1], 1] * 256
#         end_node_x = pos[edge_nodes[0], 0] * 256
#         end_node_y = 256 - pos[edge_nodes[0], 1] * 256
#         print('Start Node x: {}'.format(start_node_x))
#         print('Start Node y: {}'.format(start_node_y))
#         plt.plot([start_node_x, start_node_y], [end_node_x, end_node_y], 'ro-')
#         plt.scatter(start_node_x, start_node_y, c='red')
#         plt.scatter(end_node_x, end_node_y, c='black')
#         edge_count += 1
#         if edge_count % params.graph_k == 0:
#             # plt.scatter(start_node_x, start_node_y)
#             print('Start Node x: {}'.format(start_node_x))
#             print('Start Node y: {}'.format(start_node_y))
#             plt.show()
#         plt.scatter(x_scatter, y_scatter, c='green')

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


