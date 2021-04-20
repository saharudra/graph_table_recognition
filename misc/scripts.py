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

import torch
import torch.nn as nn
import torch.nn.functional as F

# loss = nn.BCELoss()
# target = F.softmax(torch.randn(64, 1), dim=1)
# pred = F.softmax(torch.randn(64, 1), dim=1)

# print(target.shape, pred.shape)
# out = loss(pred, target)
# print(out)

loss = nn.NLLLoss()
pred = F.log_softmax(torch.randn(5, 2), dim=1)
target = torch.tensor([1, 0, 1, 1, 0])
out = loss(pred, target)
print(out)


