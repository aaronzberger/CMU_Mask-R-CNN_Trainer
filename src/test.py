import os
import random
import time

import cv2
import numpy as np
from detectron2.data import (DatasetCatalog, MetadataCatalog,
                             build_detection_test_loader)
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import ColorMode, Visualizer

from utils import get_custom_config, get_split_paths, register_dataset

#########################################################
# PARAMETERS

num_train_images = 0
num_test_images = 10

# If you have another directory of images you want to test the network on, use this:
custom_test_dir = None  # Keep as None if you want to use the test dataset
#########################################################

register_dataset('train')
train_metadata, dataset_dicts = MetadataCatalog.get('coco-train'), DatasetCatalog.get('coco-train')

register_dataset('val')

cfg = get_custom_config(load_saved=True)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

predictor = DefaultPredictor(cfg)

times = []

# Show results for 3 random training images
for d in random.sample(dataset_dicts, num_train_images):
    im = cv2.imread(d['file_name'])
    start_time = time.time()
    outputs = predictor(im)
    times.append(time.time() - start_time)
    v = Visualizer(
        im[:, :, ::-1],
        metadata=train_metadata,
        scale=0.8,
        instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
    cv2.imshow('type: {}, path: {}'.format('TRAIN', d['file_name']), v.get_image()[:, :, ::-1])
    cv2.waitKey(delay=3000)
    cv2.destroyAllWindows()

print(times, np.average(times))

test_dir = get_split_paths('val')[1] if custom_test_dir is None else custom_test_dir

num_test_images = min(num_test_images, len(os.listdir(test_dir)))
for d in random.sample(os.listdir(test_dir), num_test_images):
    im = cv2.imread(os.path.join(test_dir, d))
    outputs = predictor(im)
    v = Visualizer(
        im[:, :, ::-1],
        metadata=train_metadata,
        scale=0.8,
        instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs['instances'].to('cpu'))
    cv2.imshow('type: {}, path: {}'.format('TEST', os.path.join(test_dir, d)), v.get_image()[:, :, ::-1])
    cv2.waitKey(delay=3000)
    cv2.destroyAllWindows()

# # Run the evaluator and print evaluation statistics on the train dataset for this model
# evaluator = COCOEvaluator('coco-train', tasks=('bbox', 'segm'), distributed=False)
# val_loader = build_detection_test_loader(cfg, 'coco-train')
# print('---------------------TRAIN---------------------')
# print(inference_on_dataset(predictor.model, val_loader, evaluator))

# Run the evaluator and print evaluation statistics on the train dataset for this model
evaluator = COCOEvaluator('coco-val', tasks=('bbox', 'segm'), distributed=False)
val_loader = build_detection_test_loader(cfg, 'coco-val')
print('---------------------TEST---------------------')
print(inference_on_dataset(predictor.model, val_loader, evaluator))
