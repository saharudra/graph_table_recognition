"""
Visualizing PubTabNet dataset's ground truth. 
"""
import json
import os
import cv2
import matplotlib.pyplot as plt

# # load json file
# root = '/Users/i23271/Downloads/table/datasets/PubTabNet/examples/'
# annotation_filename = root + 'PubTabNet_Examples.jsonl'

# annotations = []
# with open(annotation_filename, 'r') as jaf:
#     for line in jaf:
#         annotations.append(json.loads(line))

# # print(len(annotations))
# # annot = annotations[18]
# annot = annotations[4]
# # annot = annotations[13]
# # import pdb; pdb.set_trace()
# # load image
# img_filename = root + annot['filename']
# img = cv2.cvtColor(cv2.imread(img_filename), cv2.COLOR_BGR2RGB)
# cell_annotation = annot['html']['cells']
# cell_structure = annot['html']['structure']['tokens']
# print(cell_annotation)
# print(cell_structure)

# # Plot image with cell-text bounding box
# for cell in cell_annotation:
#     if 'bbox' in cell:
#         bbox = cell['bbox']
#         x0, y0, x1, y1 = bbox
#         cv2.rectangle(img, (x0, y0), (x1, y1),(0, 0, 255), 2)
# plt.imshow(img)
# plt.show()
# # import pdb; pdb.set_trace()

# Plotting updated annotations
root = '/datatop_1/rudra/table_recognition/datasets/pubtabnet/val'

filelist = os.listdir(root)
json_filelist = [i for i in filelist if i.endswith('.json')]

for file in json_filelist:
    # read image
    img_filename = os.path.join(root, file[:-4] + 'png')
    img = cv2.cvtColor(cv2.imread(img_filename), cv2.COLOR_BGR2RGB)

    with open(file, 'r') as jf:
        annot = json.load(jf)
    
    print(annot['cell_annotation'])
    print('*' * 100)
    pritn(annot['cell_structure'])
    
    for cell in annot['cell_annotation']:
        if 'bbox' in cell:
            bbox = cell['bbox']
            x0, y0, x1, y1 = bbox
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 2)
    plt.imshow(img)
    plt.show()

    import pdb; pdb.set_trace()
