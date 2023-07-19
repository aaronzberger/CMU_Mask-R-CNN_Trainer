'''Process a half-labeled dataset into a fully labeled dataset'''

import json
import sys
import os

LAST_IMAGE_INDEX = 299

if len(sys.argv) < 2:
    print('Usage: python dataset_split.py <data_dir>')
    sys.exit()
data_dir = sys.argv[1]

annotations = json.load(open(os.path.join(data_dir, 'annotations.json')))

new_annotations = annotations.copy()
new_annotations['images'] = annotations['images'][:LAST_IMAGE_INDEX + 1]

valid_filenames = [x['file_name'] for x in new_annotations['images']]
valid_filenames = [x.split('/')[-1] for x in valid_filenames]

# Remove all images after the last image index in the images/ folder
for filename in os.listdir(os.path.join(data_dir, 'images')):
    if filename not in valid_filenames:
        os.remove(os.path.join(data_dir, 'images', filename))

json.dump(new_annotations, open(os.path.join(data_dir, 'annotations.json'), 'w'))
