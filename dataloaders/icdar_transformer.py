import random
import cv2
import numpy as np
import os
from shapely.geometry import Point, Polygon

import torch
from torch_geometric.data import Data, Dataset, DataLoader
from torch_scatter import scatter_mean
import math
import json
import csv

from misc.args import scitsr_params
from ops.utils import resize_image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Icdar2013DatasetSB(Dataset):
    def __init__(self, params, partition='train', transform=None, pre_transform=None):
        super(Icdar2013DatasetSB, self).__init__(params, transform, pre_transform)

        self.params = params
        

class Icdar2019DatasetSB(Dataset):
    def __init__(self, params, partition='train', transform=None, pre_transform=None):
        super(Icdar2019DatasetSB, self).__init__(params, transform, pre_transform)

        self.params = params
        