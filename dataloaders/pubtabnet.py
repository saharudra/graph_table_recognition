import random
import cv2
import numpy as np
import os
import codecs

import torch
from torch_geometric.data import Data, Dataset, DataLoader
from torch_scatter import scatter_mean
import math
import json
import csv

from misc.args import pubtabnet_parms
from ops.utils import resize_image

class PubTabNetDataset(Dataset):
    """
    Dataset Generator for S-A and S-B experimentation regimies for transformer
    based architecture.
    S-A: Input to the model is only the image of the table. 
         Model consists of a module to extract bounding boxes
         and cluster them into cell-text bounding boxes.
         
    S-B: Input to the model is the image of the table as well as
         cell-text bounding boxes. 

    PubTabNet and SciTSR only consists of cell-text level bounding
    boxes.

    Sanity Checks:
    * Plot position features on image
    TODO: MCTS based sampling for row and col classification for pairs of cell-texts.
    """

    def __init__(self, params, partition='train', transform=None, pre_transform=None):
          super(PubTabNetDataset, self).__init__(params, transform, pre_transform)

          self.params = params
          self.partition = partition
          self.root_path = os.path.join(self.params.data_dir, self.partition)

          # Create a list of images in a json file
          self.jsonfile = os.path.join(self.root_path, 'imglist.json')
          if os.path.exists(self.jsonfile) and not self.params.new_imglist:
               with open(self.jsonfile, 'r') as rf:
               self.imglist = json.load(rf)
          else:
               self.imglist = list(filter(lambda fn: fn.lower().endswith('.png') or fn.lower().endswith('.jpg'),
                                          os.listdir(os.path.join(self.root_path, partition))))
               self.imglist = self.check_all()
               with open(self.jsonfile, 'w+') as wf:
                    json.dump(self.imglist, wf)
          
          self.img_size = self.params.img_size

     @property
     def raw_file_names(self):
          return []

     @property
     def processed_file_names(self):
          return []

     def read_structure(self):
          return 
    
     def reset(self):
          pass

     # remove empty cells
     def remove_empty_cell(self, chunks):

          new_chunks = []

          for chunk in chunks:
               if 'bbox' not in chunk:
                    continue
               else:
                    new_chunks.append(chunk)

          return new_chunks

     def readlabel(self, idx):
          """
          Returns: 
          :img: Resized image while mainitaining aspect ratio and padding with 0.
          :chunks: List of cell-text information of the following format
               [
                    {'tokens': [comma separated characters of cell text],
                     'bbox': [x0, y0, x1, y1],
                     'start_row': 0-indexed integer value,
                     'start_col': 0-indexed integer value,
                     'end_row': 0-indexed integer value,
                     'end_col': 0-indexed integer value}, ...
               ]

               Bounding boxes have been modified based on image resizing parameter.
          """
          imgfn = self.imglist[idx]

          chunkfn = os.path.join(os.path.join(self.root_path, self.partition), 
                                 os.path.splitext(os.path.basename(imgfn))[0] + '.json')
          imgfn = os.path.join(os.path.join(self.root_path, self.partition),
                               os.path.splitext(os.path.basename(imgfn))[0] + '.png')
          
          if not os.path.exists(chunkfn) os not os.path.exists(imgfn):
               print('Files not found: {}, {}'.format(chunkfn, imgfn))
               return
          
          with open(chunkfn, 'r') as cf:
               chunks = json.load(cf)['html']['cells']
          
          img = cv2.cvtColor(cv2.imread(imgfn), cv2.COLOR_BGR2RGB)
          if img is not None:
               h, w, c = img.shape
               img, window, scale, padding, crop = resize_image(img, min_dim=self.params.img_size, max_dim=self.params.img_size,
                                                                min_scale=self.params.img_scale)
               h_n, w_n, c_n = img.shape

               if w > h:
                    # width > height, offset added in height or y direction
                    # scale bbox in x direction directly
                    offset = (self.params.img_size - math.floor((self.params.img_size * h) / w)) / 2
               
               elif h > w:
                    # similarly if height > width, offset added in width or x direction
                    # scale bbox in y direction directly
                    offset = (self.params.img_size - math.floor((self.params.img_size * w) / h)) / 2
               
               else:
                    offset = 0

               # Transform chunks bounding boxes
               for cell in chunks:
                    if 'bbox' in cell:
                         bbox = cell['bbox']
                         x0, y0, x1, y1 = bbox

                         if w > h:
                              x0 = int((x0 / w) * w_n)
                              x1 = int((x1 / w) * w_n)
                              y0 = int( offset + ((y0 / h) * math.floor((self.params.img_size * h) / w)) )
                              y1 = int( offset + ((y1 / h) * math.floor((self.params.img_size * h) / w)) )

                         elif h > w:
                              x0 = int( offset + ((x0 / w) * math.floor((self.params.img_size * w) / h)) )
                              x1 = int(offset + ((x1 / w ) * math.floor((self.params.img_size * w) / h)) )
                              y0 = int((y0 / h) * h_n)
                              y1 = int((y1 / h) * h_n)
                         
                         else:
                              x0 = int((x0 / w) * w_n)
                              x1 = int((x1 / w) * w_n)
                              y0 = int((y0 / h) * h_n)
                              y1 = int((y1 / h) * h_n)

                         # Changing bbox indexs to match scitsr/icdar2013/... dataloaders.
                         cell['bbox'] = [x0, x1, y0, y1]
          
          return img, chunks


     def __len__(self):
          return len(self.imglist)
     
     def cal_chk_limits(self, chunks):

          x_min = min(chunks, key=lambda p: p['bbox'][0])['bbox'][0]
          y_min = min(chunks, key=lambda p: p['bbox'][1])['bbox'][1]
          x_max = max(chunks, key=lambda p: p['bbox'][2])['bbox'][2]
          y_max = max(chunks, key=lambda p: p['bbox'][3])['bbox'][3]
          hlist = [p['bbox'][3] - p['bbox'][1] for p in chunks]
          avg_hei = sum(hlist) / len(hlist)
          width = x_max - x_min + 2 * avhei
          height = y_max - y_min + 0.5 * 2 * avhei
          # Maintain position features index similar to scitsr/icdar2013/... dataloader
          # for carryign out transfer-learning experiments.

          return [x_min, x_max, y_min, y_max, width, height, avhei]

     def pos_feature(self, chk, cl):

          # Same as that of scitsr/icdar2013/... dataloaders as indexing has been changed.
          # See methods: cal_chk_limits, readlabel
          x1 = (chk["bbox"][0] - cl[0] + cl[6]) / cl[4]
          x2 = (chk["bbox"][1] - cl[0] + cl[6]) / cl[4]
          y1 = (chk["bbox"][2] - cl[2] + 0.5 * cl[6]) / cl[5]
          y2 = (chk["bbox"][3] - cl[2] + 0.5 * cl[6]) / cl[5]
          center_x = (x1 + x2) * 0.5  
          center_y = (y1 + y2) * 0.5
          width = x2 - x1  
          height = y2 - y1  

          return [x1, x2, y1, y2, center_x, center_y, width, height]

     def augmentation_chk(self, chunks):
          for chk in chunks:
               chk["bbox"][0] += random.normalvariate(0, 1)
               chk["bbox"][1] += random.normalvariate(0, 1)
               chk["bbox"][2] += random.normalvariate(0, 1)
               chk["bbox"][3] += random.normalvariate(0, 1)

     def cal_row_label(slef, tbpos):
          """
          Calculating for all-pairs of cell-texts. This is for the transformer based
          architecture without sampling for loss calculation.
          Here, row_classification is done for all of the cell-texts w.r.t the first cell text
          then for remaining w.r.t second cell-text and so on..
          After the transformer module, concatenation of cell-text being done in the same manner
          to maintain correspondance between input and output for each table image.
          TODO: MCTS sampling based row classification.
          """
          y = []

          for cell_i in range(len(tbpos)):
               for cell_j in range(cell_i + 1, len(tbpos)):
                    source_start, source_end = tbpos[cell_i][0], tbpos[cell_i][1]
                    target_start, target_end = tbpos[cell_j][0], tbpos[cell_j][1]
                    if (source_start >= target_start and source_end <= target_end):
                         y.append(1)
                    elif (target_start >= source_start and target_end <= source_end):
                         y.append(1)
                    else:
                         y.append(0)

          return y

     def cal_col_label(self, data, tbpos):
          """
          Calculating for all-pairs of cell-texts. This is for the transformer based
          architecture without sampling for loss calculation.
          Here, col_classification is done for all of the cell-texts w.r.t the first cell text
          then for remaining w.r.t second cell-text and so on..
          After the transformer module, concatenation of cell-text being done in the same manner
          to maintain correspondance between input and output for each table image.
          TODO: MCTS sampling based col classification.
          """
          y = []

          for cell_i in range(len(tbpos)):
               for cell_j in range(cell_i + 1, len(tbpos)):
                    source_start, source_end = tbpos[cell_i][2], tbpos[cell_i][3]
                    target_start, target_end = tbpos[cell_j][2], tbpos[cell_j][3]
                    if (source_start >= target_start and source_end <= target_end):
                         y.append(1)
                    elif (target_start >= source_start and target_end <= source_end):
                         y.append(1)
                    else:
                         y.append(0)

          return y

     def get(self, idx):
          img, chunks = self.readlabel(idx)

          if self.params.augment_chunk:
               self.augmentation_chk(chunks)

          x, pos, tbpos, imgpos, cell_wh = [], [], [], [], []

          chunks = self.remove_empty_cell(chunks)
          # Calculating chunk limits after removing empty cells
          cl = self.cal_chk_limits(chunks)

          for chunk in chunks:
               xt = self.pos_feature(chunk, cl)
               x.append(xt)
               pos.append(xt[4:6])
               tbpos.append([chunk['start_row'], chunk['end_row'], chunk['start_col'], chunk['end_col']])
               imgpos.append([(1.0 - xt[5]) * 2 - 1.0, xt[4] * 2 - 1.0])
               cell_wh.append([xt[-2], xt[-1]])

          x = torch.FloatTensor(x)
          pos = torch.FloatTensor(pos)
          data = Data(x=x, pos=pos)
          
          y_row = self.cal_row_label(tbpos)
          y_col = self.cal_col_label(tbpos)

          data.y_row = torch.LongTensor(y_row) 
          data.y_col = torch.LongTensor(y_col)
          data.img = img
          data.imgpos = torch.FloatTensor(imgpos)
          data.cell_wh = torch.FloatTensor(cell_wh)
          data.nodenum = torch.LongTensor([len(chunks)])
          
          return data
       