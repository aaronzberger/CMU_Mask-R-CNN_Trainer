import os

import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode
from tqdm import tqdm

from utils import data_path, get_custom_config, register_dataset

#########################################################
# PARAMETERS

# image_dir = os.path.join(data_path, 'test', 'images')
image_dir = '/home/frc/Downloads/corn_novel'
fps = 1  # frames per second of the video
#########################################################

register_dataset('train')
train_metadata, dataset_dicts = MetadataCatalog.get('coco-train'), DatasetCatalog.get('coco-train')

cfg = get_custom_config(load_saved=True)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

predictor = DefaultPredictor(cfg)

example_image = cv2.imread(os.path.join(image_dir, os.listdir(image_dir)[0]), cv2.IMREAD_COLOR)
height, width, channels = example_image.shape
out = cv2.VideoWriter(os.path.join(cfg.OUTPUT_DIR, 'video.avi'), cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))

filenames = sorted(os.listdir(image_dir))

v = VideoVisualizer(metadata=train_metadata, instance_mode=ColorMode.IMAGE_BW)

for filename in tqdm(iterable=filenames, total=len(filenames)):
    image = cv2.imread(os.path.join(image_dir, filename))
    outputs = predictor(image)
    visualization = v.draw_instance_predictions(image[:, :, ::-1], outputs['instances'].to('cpu'))
    visualization = cv2.cvtColor(visualization.get_image(), cv2.COLOR_RGB2BGR)
    out.write(visualization)

out.release()
