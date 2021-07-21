from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances

import os
import sys


#########################################################
# PARAMETERS

data_path = '/home/aaron/tree_image_data'
score_threshold = 0.8  # confidence score needed to be considered positive
nms_threshold = 0.1  # lower score -> less overlap required to eliminate
num_classes = 2  # background + number of labeled classes
learning_rate = 0.02
training_iterations = 100
batch_size_per_image = 128  # how many proposals to sample for loss calculation
                            # increase to 256 or 512 for slower but better performance
batch_size = 2  # number of images per training batch
#########################################################


OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'

def register_dataset(split):
    '''Register the train dataset and return the metadata and dataset catalog for Detectron2 training'''
    label_path, image_path = get_split_paths(split)
    if not os.path.exists(label_path):
            print(FAIL + 'Unable to find labels for {} at {}'.format(split, label_path) + ENDC)
            sys.exit()
    if not os.path.exists(image_path):
            print(FAIL + 'Unable to find image dir for {} at {}'.format(split, image_path) + ENDC)
            sys.exit()

    register_coco_instances('coco-{}'.format(split), {}, label_path, image_path)

def get_split_paths(split):
    label_path = os.path.join(data_path, split, '{}.json'.format(split))
    image_path = os.path.join(data_path, split, 'images')

    return label_path, image_path


def get_custom_config(load_saved=False):
    '''Load and edit the detectron2 config for the pretrained model'''
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
    )
    cfg.DATASETS.TRAIN = ('coco-train',)
    cfg.DATASETS.TEST = ()  # no metrics implemented for this dataset
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold  # set threshold for this model
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')  # initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = learning_rate
    cfg.SOLVER.MAX_ITER = (training_iterations)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (batch_size_per_image)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_threshold

    if load_saved:
        model_path = os.path.join(cfg.OUTPUT_DIR, 'model_final.pth')
        if not os.path.exists(model_path):
            print(FAIL + 'Unable to find saved model at {}'.format(model_path) + ENDC)
            sys.exit()
        cfg.MODEL.WEIGHTS = model_path
        print(OKGREEN + 'Loaded weights from {}...'.format(model_path) + ENDC)

    cfg.DATASETS.TEST = ('coco-test',)
    cfg.TEST.EVAL_PERIOD = 100

    return cfg