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

class ScitsrDatasetSB(Dataset):
    """
    SciTSR dataloader for transformer based table structure recognition.
    Graph is not being formed, position and image information is being passed 
    to the transformer.
    """
    def __init__(self, params, partition='train', transform=None, pre_transform=None):
        super(ScitsrDatasetSB, self).__init__(params, transform, pre_transform)

        self.params = params
        self.root_path = os.path.join(self.params.data_dir, partition)

        # Create a list of images as a json file
        self.jsonfile = os.path.join(self.root_path, 'imglist_sb.json')
        if os.path.exists(self.jsonfile) and not self.params.new_imglist:
            with open(self.jsonfile, 'r') as rf:
                self.imglist = json.load(rf)
        else:
            self.imglist = list(filter(lambda fn: fn.lower().endswith('.jpg') or fn.lower().endswith('.png'),
                                       os.listdir(os.path.join(self.root_path, "img"))))
            self.imglist = self.check_all()
            with open(self.jsonfile, "w+") as wf:
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

    # Check which images are valid for processing
    def check_all(self):
        validlist = []
        for idx in range(len(self.imglist)):
            print('*** file:', self.imglist[idx])
            # structs, chunks, img, rels = self.readlabel(idx)
            structs, chunks, img = self.readlabel(idx)
            print(structs, chunks)
            vi = self.check_chunks(structs, chunks)
            if vi == 1 and (img is not None):
                validlist.append(self.imglist[idx])
        print('num valid imgs:', len(validlist))
        return validlist

    # Remove empty cells i.e. cells with no text from structs
    def remove_empty_cell(self, structs):
        structs.sort(key=lambda p: p["id"])
        news = []
        idx = 0
        for st in structs:
            text = st["tex"].strip().replace(" ", "")
            if text == "" or text == '$\\mathbf{}$' or len(st["content"]) == 0:  # empty cell
                continue
            st["id"] = idx
            news.append(st)
            idx += 1
        return news

    # Checks correspondace between chunks and structs
    def check_chunks(self, structs, chunks):
        structs = self.remove_empty_cell(structs)
        # Sanity check for labeling.
        if len(structs) != len(chunks) and self.params.labeling_sanity:
            print('Fails sanity check')
            return 0
        for st in structs:
            id = st["id"]
            if id >= len(chunks):
                print("chunk index out of range.", id)
                return 0
            ch = chunks[id]
            txt1 = st["tex"].replace(" ", "")
            txt2 = ch["text"].replace(" ", "")
            if txt1 != txt2:
                print(id, "mismatch:", txt1, " ", txt2)
            if st["end_row"] - st["start_row"] != 0 or st["end_col"] - st["start_col"] != 0:
                print("span cells:", id)
        return 1

    def readlabel(self, idx):
        imgfn = self.imglist[idx]
       
        structfn = os.path.join(self.root_path, "structure", os.path.splitext(os.path.basename(imgfn))[0] + ".json")
        chunkfn = os.path.join(self.root_path, "chunk", os.path.splitext(os.path.basename(imgfn))[0] + ".chunk")
        imgfn = os.path.join(self.root_path, "img", os.path.splitext(os.path.basename(imgfn))[0] + ".png")
        
        if not os.path.exists(structfn) or not os.path.exists(chunkfn) or not os.path.exists(imgfn):
            print("can't find files.")
            return
        
        with open(chunkfn, 'r') as f:
            chunks = json.load(f)['chunks']
        if len(chunks) == 0:
            print(chunkfn)
        with open(structfn, 'r') as f:
            structs = json.load(f)['cells']
        # print(os.stat(imgfn).st_size == 0)
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

            rescale_params = [h, w, h_n, w_n, offset]

        return structs, chunks, img, rescale_params
    
    def __len__(self):
        return len(self.imglist)

    def box_center(self, chkp):
        # x1, x2, y1, y2  in chunk file
        # Returns centroid of the cell text bounding box
        return [(chkp[0] + chkp[1]) / 2, (chkp[2] + chkp[3]) / 2]

    def cal_chk_limits(self, chunks):
        x_min = min(chunks, key=lambda p: p["pos"][0])["pos"][0]
        x_max = max(chunks, key=lambda p: p["pos"][1])["pos"][1]
        y_min = min(chunks, key=lambda p: p["pos"][2])["pos"][2]
        y_max = max(chunks, key=lambda p: p["pos"][3])["pos"][3]
        hlist = [p["pos"][3] - p["pos"][2] for p in chunks]
        avhei = sum(hlist) / len(hlist)
        width = x_max - x_min + 2 * avhei
        height = y_max - y_min + 0.5 * 2 * avhei
        return [x_min, x_max, y_min, y_max, width, height, avhei]

    def pos_feature(self, chk, cl):
        x1 = (chk["pos"][0] - cl[0] + cl[6]) / cl[4]
        x2 = (chk["pos"][1] - cl[0] + cl[6]) / cl[4]
        y1 = (chk["pos"][2] - cl[2] + 0.5 * cl[6]) / cl[5]
        y2   = (chk["pos"][3] - cl[2] + 0.5 * cl[6]) / cl[5]
        center_x = (x1 + x2) * 0.5  
        center_y = (y1 + y2) * 0.5
        width = x2 - x1  
        height = y2 - y1
        return [x1, x2, y1, y2, center_x, center_y, width, height]

    def augmentation_chk(self, chunks):
        for chk in chunks:
            chk["pos"][0] += random.normalvariate(0, 1)
            chk["pos"][1] += random.normalvariate(0, 1)
            chk["pos"][2] += random.normalvariate(0, 1)
            chk["pos"][3] += random.normalvariate(0, 1)

    def transform_pos_features(self, xt, rescale_params):
        h, w, h_n, w_n, offset = rescale_params
        x1, x2, y1, y2, center_x, center_y, width, height = xt

        if w > h:
            y1 = (self.params.img_size - y1 * (self.params.img_size - 2 * offset) - offset) / self.params.img_size
            y2 = (self.params.img_size - y2 * (self.params.img_size - 2 * offset) - offset) / self.params.img_size
            center_y = (y1 + y2) * 0.5

        elif h > w:
            x1 = (x1 * (self.params.img_size - 2 * offset) - offset) / self.params.img_size
            x2 = (x2 * (self.params.img_size - 2 * offset) - offset) / self.params.img_size
            center_x = (x1 + x2) * 0.5
        else:
            y1 = 1 - y1
            y2 = 1 - y2
            center_y = (y1 + y2) * 0.5

        xt = [x1, x2, y1, y2, center_x, center_y, width, height]

        return xt

    def get(self, idx):
        # structs, chunks, img, scaling_params = self.readlabel(idx)
        structs, chunks, img, rescale_params = self.readlabel(idx)

        if self.params.augment_chunk:
            self.augmentation_chk(chunks)

        cl = self.cal_chk_limits(chunks)

        # Transform scale of bounding boxes
        # SciTSR has bounding box information w.r.t the pdf but doesn't have the exact location
        # on the pdf. Transform x_min and y_min to 0, 0.
        # chunks = self.transform_chk_bbox(chunks, cl, scaling_params)
        # cl = self.cal_chk_limits(chunks)
        # print(cl)
        x, pos, tbpos, imgpos, cell_wh = [], [], [], [], []
        # offset = []
        structs = self.remove_empty_cell(structs)

        for st in structs:
            id = st["id"]
            chk = chunks[id]
            xt = self.pos_feature(chk, cl)
            xt = self.transform_pos_features(xt, rescale_params)
            x.append(xt)
            pos.append(xt[4:6])  # pos only takes the centroid of each cell text bounding box
            tbpos.append([st["start_row"], st["end_row"], st["start_col"], st["end_col"]])  # position information in the table to calculate label
            imgpos.append([(1.0 - xt[5]) * 2 - 1.0, xt[4] * 2 - 1.0])
            cell_wh.append([xt[-2], xt[-1]])
            # offset.append(chk['offset'])

        x = torch.FloatTensor(x)
        pos = torch.FloatTensor(pos)
        data = Data(x=x, pos=pos)

        y_row = self.cal_all_pair_row_label(data, tbpos)
        y_col = self.cal_all_pair_col_label(data, tbpos)

        img = torch.FloatTensor(img / 255.0).permute(2, 0, 1).unsqueeze(0)

        data.y_row = torch.LongTensor(y_row)
        data.y_col = torch.LongTensor(y_col)
        data.img = img
        data.imgpos = torch.FloatTensor(imgpos)
        data.cell_wh = torch.FloatTensor(cell_wh)
        data.nodenum = torch.LongTensor([len(structs)])

        return data

    def cal_all_pair_row_label(self, data, tbpos):
        y = []
        for si in range(data.x.shape[0]):
            for ei in range(si + 1, data.x.shape[0]):
                y.append(self.is_same_row(si, ei, tbpos))
        return y

    def cal_all_pair_col_label(self, data, tbpos):
        y = []
        for si in range(data.x.shape[0]):
            for ei in range(si + 1, data.x.shape[0]):
                y.append(self.is_same_col(si, ei, tbpos))
        return y

    def is_same_row(self, si, ei, tbpos):
        ss, se = tbpos[si][0], tbpos[si][1]
        ts, te = tbpos[ei][0], tbpos[ei][1]
        if (ss >= ts) and (se <= te):
            return 1
        if (ts >= ss) and (te <= se):
            return 1
        return 0

    def is_same_col(self, si, ei, tbpos):
        ss, se = tbpos[si][2], tbpos[ei][3]
        ts, te = tbpos[ei][2], tbpos[ei][3]
        if (ss >= ts) and (se <= te):
            return 1
        if (ts >= ss) and (te <= se):
            return 1
        return 0

    
if __name__ == '__main__':

    from misc.args import *
    
    params = scitsr_params()
    train_dataset = ScitsrDatasetSB(params)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    print(params)

    for idx, data in enumerate(train_loader):
        print(data)
        import pdb; pdb.set_trace()
