#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.
"""

import os
import argparse

import cv2

import rosbag
from cv_bridge import CvBridge
from detectron2.utils.visualizer import ColorMode
import cv2 as cv
import numpy as np
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import MetadataCatalog
from tqdm import tqdm


class MaskRCNN:
    def __init__(self):
        self.model_cfg = get_cfg()
        self.model_cfg.merge_from_file(
            model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        )
        self.model_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
        self.model_cfg.MODEL.WEIGHTS = '/home/frc/catkin_ws/src/stalk_detect/model_field_day1.pth'
        # self.model_cfg.MODEL.WEIGHTS = '/home/frc/Documents/CMU_Mask-R-CNN_Trainer/output/1epoch.pth'
        self.model_cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # background, stalk
        self.model_cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.1
        self.model_cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = (128)
        self.model_cfg.MODEL.DEVICE = 'cuda:0'

        self.model = DefaultPredictor(self.model_cfg)

    def forward(self, image):
        '''
        Forward Prop for the Mask-R-CNN model

        Parameters
            image (cv.Mat/np.ndarray): the input image

        Returns
            np.ndarray: confidence scores for each prediction
            np.ndarray: (N X 4) array of bounding box corners
            np.ndarray: (N X H X W) masks for each prediction,
                where the channel 0 index corresponds to the prediction index
                and the other channels are boolean values representing whether
                that pixel is inside that prediction's mask
            dict: output object from detectron2 predictor
        '''
        outputs = self.model(image)

        scores = outputs['instances'].scores.to('cpu').numpy()

        bboxes = outputs['instances'].pred_boxes.tensor.to('cpu').numpy()
        masks = outputs['instances'].pred_masks.to('cpu').numpy()

        # Convert xyxy corner format to xywh center format
        bboxes_center = [[(i[0] + i[2]) / 2, (i[1] + i[3]) / 2, abs(i[2] - i[0]), abs(i[3] - i[1])] for i in bboxes]

        return scores, bboxes_center, masks, outputs

    def visualize(self, input_image, output):
        '''
        Visualize the results of the model

        Parameters
            image (cv.Mat/np.ndarray): the input image
            output (dict): output from the detectron2 predictor

        Returns
            np.ndarray: visualized results
        '''

        v = Visualizer(input_image[:, :, ::-1],
                       metadata=MetadataCatalog.get('empty'),
                       instance_mode=ColorMode.IMAGE)  # remove the colors of unsegmented pixels

        v = v.draw_instance_predictions(output['instances'].to('cpu'))

        return v.get_image()[:, :, ::-1]


def run_detection(cls, cv_image):
    '''
    Run the Mask R-CNN model on the given image

    Parameters
        cv_image (np.ndarray): The image to run the model on

    Returns
        masks (np.ndarray): The masks of the detected stalks
    '''
    # Run the model
    scores, bboxes, masks, output = cls.model.forward(cv_image)
    masks = masks.astype(np.uint8) * 255

    return masks, output


def main():
    """Extract a folder of images from a rosbag.
    """
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("output_dir", help="Output directory.")
    parser.add_argument("image_topic", help="Image topic.")
    parser.add_argument("only_every_nth", help="Only extract every nth image.", type=int, default=1)

    args = parser.parse_args()

    print("Extract images from {} on topic {} into {}".format(
        args.bag_file, args.image_topic, args.output_dir))

    model = MaskRCNN()

    bag = rosbag.Bag(args.bag_file, "r")
    bridge = CvBridge()
    count = 0
    for topic, msg, t in tqdm(bag.read_messages(topics=[args.image_topic]), total=bag.get_message_count(args.image_topic),
                              desc="Extracting images", unit="images", colour="green"):
        if count % args.only_every_nth != 0:
            count += 1
            continue
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        scores, centers, masks, output = model.forward(cv_image)
        print(scores)
        features_image = model.visualize(cv_image, output)

        cv2.imwrite(os.path.join(args.output_dir, "frame%06i.png" % count), features_image)

        count += 1

    bag.close()


if __name__ == '__main__':
    main()
