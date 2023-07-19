'''Ensure every image in a file is horizontal'''

import os
import sys

import cv2
from tqdm import tqdm

if len(sys.argv) != 2:
    print('Usage: python ensure_horizontal.py <input_dir>')
    sys.exit(1)
input_dir = sys.argv[1]

for image_name in tqdm(os.listdir(input_dir)):
    image_path = os.path.join(input_dir, image_name)

    # Check if the image is horizontal
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    if height > width:
        # Rotate the image
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        cv2.imwrite(image_path, image)
