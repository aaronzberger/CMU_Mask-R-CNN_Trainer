'''Remove the 'images/' prefix in the labels, since Detectron2 expects only the filename'''

import json
import os

from utils import data_path

train_labels_path = os.path.join(data_path, 'train', 'train.json')
test_labels_path = os.path.join(data_path, 'test', 'test.json')

train_labels = json.load(open(train_labels_path))
test_labels = json.load(open(test_labels_path))

for label in train_labels['images']:
    label['file_name'] = label['file_name'].split('/')[-1]

for label in test_labels['images']:
    label['file_name'] = label['file_name'].split('/')[-1]

# Write the new labels to the same file
json.dump(train_labels, open(train_labels_path, 'w'))
json.dump(test_labels, open(test_labels_path, 'w'))
