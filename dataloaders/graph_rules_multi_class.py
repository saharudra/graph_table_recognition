import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import random
import cv2
import numpy as np
import os

import torch
import torch.nn as nn
import math
import json

from torch_geometric.data import Data, Dataset, DataLoader
from misc.args import scitsr_params
from ops.rules import GraphRules
from ops.utils import resize_image

class GraphRulesMultiClass(Dataset):
    """
    For each edge of the graph, ground truth classifies it as
    not same row\col  --> [0, 0, 1]
    same row only     --> [0, 1, 0]
    same col only     --> [1, 0, 0]
    An edge can belong to only one of the 3 classes.
    """
    def __init__(self, params, partition='train', transform=None, pre_transform=None):
        super(GraphRulesMultiClass, self).__init__()

        self.params = params
        self.rules = GraphRules(self.params)
        self.root_path = os.path.join(self.params.data_dir, partition)

        # Create a list of images as a json file
        self.jsonfile = os.path.join(self.root_path, 'imglist_rules.json')

        if os.path.exists(self.jsonfile) and not self.params.new_imglist:
            with open(self.jsonfile, 'r') as rf:
                self.imglist = json.load(rf)
        else:
            self.imglist = list(filter(lambda fn: fn.lower().endswith('.jpg') or fn.lower().endswith('.png'),
                                       os.listdir(os.path.join(self.root_path, "img"))))
            self.imglist = self.check_all()
            with open(self.jsonfile, "w+") as wf:
                json.dump(self.imglist, wf)
    
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
            structs, chunks, img, rescale_params = self.readlabel(idx)
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
        try:
            img = cv2.cvtColor(cv2.imread(imgfn), cv2.COLOR_BGR2RGB)
        except:
            img = None
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
        else:
            rescale_params = None
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

    def __getitem__(self, idx):
        # structs, chunks, img, scaling_params = self.readlabel(idx)
        structs, chunks, img, rescale_params = self.readlabel(idx)

        if self.params.augment_chunk:
            self.augmentation_chk(chunks)

        cl = self.cal_chk_limits(chunks)

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


        x = torch.FloatTensor(x)
        pos = torch.FloatTensor(pos)
        img = torch.FloatTensor(img / 255.0).permute(2, 0, 1).unsqueeze(0)
        # Obtain edges from set creation function using positions
        row_edge_index, col_edge_index = self.rule_based_set_generation(pos, img)
        
        # From row_edge_index and col_edge_index get all edges and their label tuples 
        # for (not same row or col, same row, same col) using self._if_same_row and 
        # self._if_same_col

        if self.params.gr_multi_class:
            # Create multi-class dataset
            edge_dict = {}

            for edge in row_edge_index:
                edge_key = str(edge[0]) + ':' + str(edge[1])
                if edge_key not in edge_dict:
                    if self._if_same_row(edge[0], edge[1], tbpos):
                        edge_tuple = [0, 1, 0]
                    elif self._if_same_col(edge[0], edge[1], tbpos):
                        edge_tuple = [1, 0, 0]
                    else:
                        edge_tuple = [0, 0, 1]
                    edge_dict[edge_key] = edge_tuple
            
            for edge in col_edge_index:
                edge_key = str(edge[0]) + ':' + str(edge[1])
                if edge_key not in edge_dict:
                    if self._if_same_row(edge[0], edge[1], tbpos):
                        edge_tuple = [0, 1, 0]
                    elif self._if_same_col(edge[0], edge[1], tbpos):
                        edge_tuple = [1, 0, 0]
                    else:
                        edge_tuple = [0, 0, 1]
                    edge_dict[edge_key] = edge_tuple
            
            edge_index = []
            edge_gt = []
            for edge_key in edge_dict.keys():
                edge_tuple = edge_dict[edge_key]
                edge_key = [int(x) for x in edge_key.split(':')]
                edge_index.append(edge_key)
                edge_gt.append(edge_tuple)

            edge_index = torch.from_numpy(np.asarray(edge_index)).t().contiguous().long()
            edge_gt = torch.from_numpy(np.asarray(edge_gt)).t().contiguous().long().permute(1, 0)

            data = Data(x=x, pos=pos, img=img, edge_index=edge_index)

            data.gt = torch.LongTensor(edge_gt)
            data.imgpos = torch.FloatTensor(imgpos)
            data.cell_wh = torch.FloatTensor(cell_wh)
            data.nodenum = torch.LongTensor([len(structs)])

            return data

    def rule_based_set_generation(self, pos, img):
        if self.params.rules_constraint == 'naive_gaussian':
            return self.rules.naive_gaussian(pos, img)

    def _if_same_row(self, si, ei, tbpos):
        ss, se = tbpos[si][0], tbpos[si][1]
        es, ee = tbpos[ei][0], tbpos[ei][1]
        if (ss >= es and se <= ee):
            return 1
        if (es >= ss and ee <= se):
            return 1
        return 0

    def _if_same_col(self, si, ei, tbpos):
        ss, se = tbpos[si][2], tbpos[si][3]
        es, ee = tbpos[ei][2], tbpos[ei][3]
        if (ss>= es and se <= ee):
            return 1
        if (es >= ss and ee <= se):
            return 1
        return 0


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from misc.args import *
    
    params = scitsr_params()
    print(params)
    val_dataset = GraphRulesMultiClass(params, partition='train')
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    
    for idx, data in enumerate(val_loader):
        print(data)
        import pdb; pdb.set_trace()

