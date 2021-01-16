"""
Visualizing PubTabNet dataset's ground truth. 
"""
import json
import cv2

# load json file
root = '/datatop_1/rudra/table_recognition/datasets/pubtabnet/'
annotation_filename = root + 'PubTabNet_2.0.0.jsonl'

annotations = json.load(annotation_filename)
print(annotations)

