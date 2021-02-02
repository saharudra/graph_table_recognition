import random
import cv2
import numpy as np
import os
import codecs
from shapely.geometry import Point, Polygon

import torch
from torch_geometric.data import Data, Dataset, DataLoader
from torch_scatter import scatter_mean
import math
import json
import csv

from misc.args import pubtabnet_parms

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
    """

    def __init__(self, params, partition='train', transform=None, pre_transform=None):
        super(PubTabNetDataset, self).__init__(params, transform, pre_transform)

        self.params = params
        self.root_path = os.path.join(self.params.data_dir, partition)

        # Create a list of images in a json file
        self.jsonfile = os.path.join