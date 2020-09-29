import random
import cv2
import numpy as np
import os
import codecs
from shapely.geometry import Point, Polygon

import torch
from torch_geometric.data import Data, Dataset, DataLoader
from torch_scatter import scatter_mean
import torch_geometric.transforms as GT
import math
import json
import csv

from misc.args import scitsr_params

alphabet = "0123456789abcdefghijklmnopqrstuvwxyz,.*# "
vob = {x:ind for ind, x in enumerate(alphabet)}

def encode_text(ins, vob, max_len = 10, default = " "):
    out = []
    sl = len(ins)
    minl = min(sl, max_len)
    for i in range(minl):
        char = ins[i].lower()  # converting to lowercase
        if char in vob:
            out.append(vob[char])
        else:
            out.append(vob[default])
    # Append default to make text length equal
    if len(out)<=max_len:
        out = out +[vob[default]]*(max_len-len(out))
    return out


class ScitsrDataset(Dataset):
    def __init__(self, params, partition='train', transform=None, pre_transform=None):
        super(ScitsrDataset, self).__init__(params, transform, pre_transform)

        self.params = params
        self.root_path = os.path.join(self.params.data_dir, partition)

        # Create a list of images as a json file
        self.jsonfile = os.path.join(self.root_path, 'imglist.json')
        if os.path.exists(self.jsonfile):
            with open(self.jsonfile, 'r') as rf:
                self.imglist = json.load(rf)
        else:
            self.imglist = list(filter(lambda fn: fn.lower().endswith('.jpg') or fn.lower().endswith('.png'),
                                       os.listdir(os.path.join(self.root_path, "img"))))
            self.imglist = self.check_all()
            with open(self.jsonfile, "w+") as wf:
                json.dump(self.imglist, wf)
        
        self.img_size = self.params.img_size
        self.kernel = np.ones((self.params.kernel_size, self.params.kernel_size), np.uint8)

        self.graph_transform = GT.KNNGraph(k=self.params.graph_k)
    
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
            structs, chunks, img, rels = self.readlabel(idx)
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
            if text == "" or text == '$\\mathbf{}$':  # empty cell
                continue
            st["id"] = idx
            news.append(st)
            idx += 1
        return news

    # Checks correspondace between chunks and structs
    def check_chunks(self, structs, chunks):
        structs = self.remove_empty_cell(structs)
        # Sanity check for labeling.
        if len(structs) != len(chunks):
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
        relfn = os.path.join(self.root_path, "rel", os.path.splitext(os.path.basename(imgfn))[0] + ".rel")
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
        img = cv2.imread(imgfn)
        if img is not None:
            # Using RGB image.
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # TODO: Test with grayscale only
            #  img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if self.params.dilate:
                img = cv2.dilate(img, self.kernel, iterations=1)
            if self.params.erode:
                img = cv2.erode(img, self.kernel, iterations=1) # To thicken lines and text..
            img = cv2.resize(img, (self.params.img_size, self.params.img_size), interpolation=cv2.INTER_AREA)

        with open(relfn, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            rels = list(reader)
        
        for idx, rel in enumerate(rels):
            rel[-1] = rel[-1][0]
            rel = [int(x) for x in rel]
            rels[idx] = rel

        return structs, chunks, img, rels
    
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
        x3 = (chk["pos"][2] - cl[2] + 0.5 * cl[6]) / cl[5]
        x4 = (chk["pos"][3] - cl[2] + 0.5 * cl[6]) / cl[5]
        x5 = (x1 + x2) * 0.5  
        x6 = (x3 + x4) * 0.5
        x7 = x2 - x1  
        x8 = x4 - x3  
        return [x1, x2, x3, x4, x5, x6, x7, x8]

    def augmentation_chk(self, chunks):
        for chk in chunks:
            chk["pos"][0] += random.normalvariate(0, 1)
            chk["pos"][1] += random.normalvariate(0, 1)
            chk["pos"][2] += random.normalvariate(0, 1)
            chk["pos"][3] += random.normalvariate(0, 1)

    def get(self, idx):
        
        structs, chunks, img, rels = self.readlabel(idx)
        
        if self.params.augment_chunk:
            self.augmentation_chk(chunks)

        cl = self.cal_chk_limits(chunks)
        
        x, pos, tbpos, xtext, imgpos = [], [], [], [], []
        plaintext = []
        structs = self.remove_empty_cell(structs)
        
        # Sanity check for labeling.
        if len(structs) != len(chunks):
            print("Err: len(struct) = {}; len(chunks) = {}".format(len(structs), len(chunks)))
            print(self.imglist[idx])
            exit(0)
        
        for st in structs:
            id = st["id"]
            chk = chunks[id]
            xt = self.pos_feature(chk, cl)
            x.append(xt)
            pos.append(xt[4:6])  # pos only takes the centroid of each cell text bounding box
            tbpos.append([st["start_row"], st["end_row"], st["start_col"], st["end_col"]])  # position information in the table to calculate label
            xtext.append(encode_text(chk["text"], vob, self.params.text_encode_len))
            plaintext.append(chk["text"].encode('utf-8'))
            imgpos.append([(1.0 - xt[5]) * 2 - 1.0, xt[4] * 2 - 1.0])  

        x = torch.FloatTensor(x)
        pos = torch.FloatTensor(pos)
        data = Data(x=x, pos=pos)
        data = self.graph_transform(data.to(self.params.device))

        y_row = self.cal_row_label(data, tbpos)
        y_col = self.cal_col_label(data, tbpos)
        img = torch.FloatTensor(img / 255.0).unsqueeze(0).unsqueeze(0)
        rels = torch.LongTensor(rels)

        data.y_row = torch.LongTensor(y_row)
        data.y_col = torch.LongTensor(y_col)
        data.img = img
        data.rels = rels
        data.imgpos = torch.FloatTensor(imgpos)
        data.nodenum = torch.LongTensor([len(structs)])
        data.xtext = torch.LongTensor(xtext)

        return data

    def cal_row_label(self, data, tbpos):  
        edges = data.edge_index  
        y = []
        for i in range(edges.size()[1]):
            y.append(self.if_same_row(edges[0, i], edges[1, i], tbpos))
        return y

    def cal_col_label(self, data, tbpos):
        edges = data.edge_index
        y = []
        for i in range(edges.size()[1]):
            y.append(self.if_same_col(edges[0, i], edges[1, i], tbpos))
        return y

    def if_same_row(self, si, ti, tbpos):
        ss, se = tbpos[si][0], tbpos[si][1]
        ts, te = tbpos[ti][0], tbpos[ti][1]
        if (ss >= ts and se <= te):
            return 1
        if (ts >= ss and te <= se):
            return 1
        return 0

    def if_same_col(self, si, ti, tbpos):
        ss, se = tbpos[si][2], tbpos[si][3]
        ts, te = tbpos[ti][2], tbpos[ti][3]
        if (ss >= ts and se <= te):
            return 1
        if (ts >= ss and te <= se):
            return 1
        return 0

    def if_same_cell(self):
        pass


if __name__ == '__main__':
    
    from misc.args import *

    params = scitsr_params()
    print(params)
    params.optimal_k_chk = True
    print(params)
    train_dataset = ScitsrDataset(params)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

    for idx, data in enumerate(train_loader):
        print(data)
        import pdb; pdb.set_trace()