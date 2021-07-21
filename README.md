# Mask-R-CNN Detector

Run a custom Mask-R-CNN model using the Detectron2 library.

## Installation
Install Detectron2 [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html).

This code is tested using the following package versions (with Detectron2 installed from source):
 - python==3.6.9
 - torch==1.8.0
 - detectron2==0.4.1
 - CUDA==11.3 (installed separately, other versions will work)

By default, this repo uses the `mask_rcnn_R_50_FPN_3x` pretrained model installed with Detectron2.

## Data Preparation

Split your images into a `train` and a `test` folder.

### Labeling
For this library, labels should be in the COCO format.

Use this free [labeling tool](https://github.com/jsbroks/coco-annotator) to annotate your images.

Feel free to label any number of classes. Just remember to update the `num_classes` variable in `utils.py` to reflect.

For both the `train` and `test` images, export the labels to a json file. Name them `train.json` and `test.json`.

### Organization

Organize your data like this:

    your_data_path  
    │  
    ├── train
    │    ├── train.json  --> Labels for the training dataset  
    │    └── images
    │        ├── image_0.png  --> Same images as passed into the COCO labeler
    │        ├── image_1.png
    │        ├── image_2.png 
    │        └── ...
    │  
    └── test
        ├── test.json  --> Labels for the testing dataset  
        └── images
            ├── image_0.png  --> Same images as passed into the COCO labeler
            ├── image_1.png
            ├── image_2.png 
            └── ...

In utils.py, change the `data_path` variable to the base path of your data (as described above), `your_data_path`.

## Training
Before training your model, edit hyperparameters in `utils.py`. Specifically, you may wish to adjust `max_iteration` to train for longer or shorter.

To train your model, run `train.py`, which will print out periodic training updates.

To confirm your data labeling and organization was successful, check the output at the beginning of running `train.py`. It will print how many images it loaded, how many it removed, and how many total class instances it found.

After completion, a metrics graph will appear, and the final model will be saved to the output directory, which will be placed from wherever you ran the file.

## Testing
To test your model, use `test.py`.

This script will display results for a specified number of random training images and testing images. These numbers can be changed in the parameters section at the top of the `test.py` file.

It will then display [COCO evaluation statistics](https://cocodataset.org/#detection-eval) for the training and testing datasets.

If you wish to test the network on a different image directory, set `custom_test_dir` to the directory path in the parameters section at the top of the `test.py` file.

## Video Visualization
If you wish to create a video out of the detections for visualization, use `test_video.py`.

Create a directory of images where the image names are ordered (perhaps name the images 0.png, 1.png, etc.). 

Edit the name of this directory and the desired fps of the video in the parameters section at the top of the `test_video.py` file.

Run `test_video.py` to generate the video.

## ROS
I built a ROS node that uses the model created here and publishes results. See [here](https://github.com/aaronzberger/CMU_Mask-R-CNN_ROS).

## Help
Please email me (aaronzberger@gmail.com) with any questions.