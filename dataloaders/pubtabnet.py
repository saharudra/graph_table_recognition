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
    Dataset Generator for S-A and S-B experimentation regimies.
    S-A: Input to the model is only the image of the table. 
         Model consists of a module to extract bounding boxes
         and cluster them into cell-text bounding boxes.
         
    S-B: Input to the model is the image of the table as well as
         cell-text bounding boxes. 

    PubTabNet and SciTSR only consists of cell-text level bounding
    boxes.

    Sanity Checks:
    * Plot position features on image
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
                    # lll'y if height > width, offset added in width or x direction
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

                         cell['bbox'] = [x0, y0, x1, y1]
          
          return img, chunks


     def __len__(self):
          return len(self.imglist)
     
     def cal_chk_limits(self, chunks):
          x_min = 
               
                    





     
       