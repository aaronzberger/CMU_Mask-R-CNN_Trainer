from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator
import logging
from collections import OrderedDict
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper, build_detection_train_loader

import os
import sys


#########################################################
# PARAMETERS

data_path = '/home/frc/Documents/iowa_data'
score_threshold = 0.8  # confidence score needed to be considered positive
nms_threshold = 0.1  # lower score -> less overlap required to eliminate
num_classes = 2  # background + number of labeled classes
learning_rate = 0.00025
training_iterations = 80000
batch_size_per_image = 128  # proposals to sample for loss calculation: increase for slower but better performance
batch_size = 4  # number of images per training batch
OUTPUT_PATH = 'model_field_day1.pth'
#########################################################


OKGREEN = '\033[92m'
WARNING = '\033[93m'
FAIL = '\033[91m'
ENDC = '\033[0m'


class Trainer(DefaultTrainer):
    '''Custom trainer to use a different learning rate and number of iterations'''
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        output_folder = os.path.join(cfg.OUTPUT_DIR, 'inference')
        return COCOEvaluator(dataset_name, cfg, False, output_dir=output_folder)

    # @classmethod
    # def test_with_TTA(cls, cfg, model):
    #     '''Test with test-time augmentation'''
    #     logger = logging.getLogger(__name__)
    #     logger.info('Running inference with test-time augmentation ...')
    #     model = GeneralizedRCNNWithTTA(cfg, model)
    #     evaluators = [
    #         cls.build_evaluator(
    #             cfg, name
    #         ) for name in cfg.DATASETS.TEST
    #     ]
    #     res = cls.test(cfg, model, evaluators)
    #     res = OrderedDict({k + '_TTA': v for k, v in res.items()})
    #     return res

    @classmethod
    def build_train_loader(cls, cfg):
        '''Add data augmentation to the training loader'''
        augs = [
            T.RandomBrightness(0.3, 1.8),
            T.RandomContrast(0.6, 1.4),
            T.RandomSaturation(0.6, 1.4),
            T.RandomLighting(0.7),
            T.RandomRotation(angle=[-20, 20], expand=False, center=None, sample_style='range'),
        ]
        return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=augs))


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
    cfg.DATASETS.TEST = ('coco-test',)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold  # set threshold for this model
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')  # initialize from model zoo
    cfg.SOLVER.LR_SCHEDULER_NAME = 'WarmupMultiStepLR'
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = learning_rate
    cfg.SOLVER.MAX_ITER = (training_iterations)
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (batch_size_per_image)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_threshold
    cfg.TEST.EVAL_PERIOD = 250
    cfg.SOLVER.CHECKPOINT_PERIOD = 5000

    if load_saved:
        model_path = os.path.join(cfg.OUTPUT_DIR, OUTPUT_PATH)
        if not os.path.exists(model_path):
            print(FAIL + 'Unable to find saved model at {}'.format(model_path) + ENDC)
            sys.exit()
        cfg.MODEL.WEIGHTS = model_path
        print(OKGREEN + 'Loaded weights from {}...'.format(model_path) + ENDC)

    return cfg
