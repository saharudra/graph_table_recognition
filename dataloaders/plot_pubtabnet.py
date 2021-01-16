"""
Visualizing PubTabNet dataset's ground truth. 
"""
import json
import cv2
import matplotlib.pyplot as plt

# load json file
root = '/datatop_1/rudra/table_recognition/PubTabNet/examples/'
annotation_filename = root + 'PubTabNet_Examples.jsonl'

annotations = []
with open(annotation_filename, 'r') as jaf:
    for line in jaf:
        annotations.append(json.loads(line))

print(len(annotations))
# import pdb; pdb.set_trace()
# load image
img_filename = root + annotations[0]['filename']
img = cv2.cvtColor(cv2.imread(img_filename), cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.show()