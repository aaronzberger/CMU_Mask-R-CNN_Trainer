'''Given train and test annotation files, split the images accordingly'''

import json
import sys
import os

if len(sys.argv) < 2:
    print('Usage: python dataset_split.py <data_dir>')
    sys.exit()
data_dir = sys.argv[1]

train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

os.makedirs(train_dir, exist_ok=False)
os.makedirs(test_dir, exist_ok=False)
os.makedirs(os.path.join(train_dir, 'images'), exist_ok=False)
os.makedirs(os.path.join(test_dir, 'images'), exist_ok=False)

train_annotation_file = os.path.join(data_dir, 'train.json')
test_annotation_file = os.path.join(data_dir, 'test.json')

image_dir = os.path.join(data_dir, 'images')

with open(train_annotation_file, 'r') as f:
    train_annotation = json.load(f)

with open(test_annotation_file, 'r') as f:
    test_annotation = json.load(f)

train_images = train_annotation['images']
test_images = test_annotation['images']

train_image_filenames = [x['file_name'] for x in train_images]
test_image_filenames = [x['file_name'] for x in test_images]

for filename in train_image_filenames:
    os.rename(os.path.join(data_dir, filename), os.path.join(train_dir, filename))

for filename in test_image_filenames:
    os.rename(os.path.join(data_dir, filename), os.path.join(test_dir, filename))

os.rename(train_annotation_file, os.path.join(train_dir, 'train.json'))
os.rename(test_annotation_file, os.path.join(test_dir, 'test.json'))

print('Split dataset into train and test sets.')
