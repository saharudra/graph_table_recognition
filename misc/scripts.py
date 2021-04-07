"""
Miscllaneous scripts! Scrub them later.
"""
# Script removing empty img files and their corresponding chunk, structure and rel
# files for SciTSR dataset.
# import os
# root = '/Users/i23271/Downloads/table/datasets/SciTSR/train'
# img_root = root + '/img'
# files = os.listdir(img_root)

# filelist = []

# for f in files:
#     filename = os.path.join(img_root, f)
#     if os.stat(filename).st_size == 0:
#         filelist.append(filename)

# print(filelist)
# print(len(filelist))

# for f in filelist:
#     fname, ext = os.path.splitext(os.path.basename(f))
#     cfn = os.path.join(os.path.join(root, 'chunk'), fname + '.chunk')
#     sfn = os.path.join(os.path.join(root, 'structure'), fname + '.json')
#     rfn = os.path.join(os.path.join(root, 'rel'), fname + '.rel')
#     pfn = os.path.join(os.path.join(root, 'pdf'), fname + '.pdf')

#     f_lst = [f, cfn, sfn, rfn, pfn]

#     for i in f_lst:
#         os.remove(i)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# loss = nn.BCELoss()
# target = F.softmax(torch.randn(64, 1), dim=1)
# pred = F.softmax(torch.randn(64, 1), dim=1)

# print(target.shape, pred.shape)
# out = loss(pred, target)
# print(out)

# Copy SciTSR images with only 25 or less total tokens
# from torch.utils.data import Dataset, DataLoader 
# from dataloaders.scitsr_loader import ScitsrDatasetSB
# from misc.args import *

# import os
# import shutil

# dst_root = '/data/rudra/table_structure_recognition/datasets/SciTSR_25/train/'

# params = scitsr_params()
# train_dataset = ScitsrDatasetSB(params)
# train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
# curtailed_dataset_fn = []
# for idx, sample in enumerate(train_loader):
#     sample = sample
#     if sample['chunk_len'][0] <= 25:
#         curtailed_dataset_fn.append(sample['imgfn'])
#     if len(curtailed_dataset_fn) > 99:
#         break

# for fil in curtailed_dataset_fn:
#     imgfn = fil[0]
#     imgfn_split = imgfn.split('/')
#     chunkfn = '/'.join(imgfn_split[:-2]) + '/chunk/' + '.'.join(imgfn_split[-1].split('.')[:-1]) + '.chunk'
#     structurefn = '/'.join(imgfn_split[:-2]) + '/structure/' + '.'.join(imgfn_split[-1].split('.')[:-1]) + '.json'
#     pdffn =  '/'.join(imgfn_split[:-2]) + '/pdf/' + '.'.join(imgfn_split[-1].split('.')[:-1]) + '.pdf'
#     relfn =  '/'.join(imgfn_split[:-2]) + '/rel/' + '.'.join(imgfn_split[-1].split('.')[:-1]) + '.rel'

#     if os.path.exists(imgfn):
#         shutil.copy(imgfn, dst_root + '/img/')
#     else:
#         print("dst error")

#     if os.path.exists(chunkfn):
#         shutil.copy(chunkfn, dst_root + '/chunk/')
#     else:
#         print("dst error")
    
#     if os.path.exists(structurefn):
#         shutil.copy(structurefn, dst_root + '/structure/')
#     else:
#         print("dst error")
    
#     if os.path.exists(relfn):
#         shutil.copy(relfn, dst_root + '/rel/')
#     else:
#         print("dst error")

#     if os.path.exists(pdffn):
#         shutil.copy(pdffn, dst_root + '/pdf/')
#     else:
#         print("dst error")
#     import pdb; pdb.set_trace()