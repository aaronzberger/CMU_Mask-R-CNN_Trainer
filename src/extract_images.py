'''Extract images from a directory of video files and save them to a directory'''

import os
import sys

import cv2
from matplotlib import pyplot as plt
from tqdm import tqdm

video_fps = 30


if len(sys.argv) != 3:
    print('Usage: python extract_images.py <input_dir> <output_dir>')
    sys.exit(1)
input_dir = sys.argv[1]
output_dir = sys.argv[2]

os.makedirs(output_dir, exist_ok=True)

image_names = [os.path.splitext(filename)[0] for filename in os.listdir(output_dir)]
already_processed_videos = set([path.split('frame')[0][:-1] for path in image_names])


def extract_images(video_name, fps=1):
    '''
    Extract images from a video at a given fps and save them to output_dir

    Parameters:
        video_name (str): Name of the video file
        fps (int): Frames per second of the video

    Returns:
        int: Number of images extracted
    '''
    video_path = os.path.join(input_dir, video_name)
    video = cv2.VideoCapture(video_path)

    # video_fps = video.get(cv2.CAP_PROP_FPS)
    frame_skip = int(video_fps / float(fps))
    num_frames_to_process = video.get(cv2.CAP_PROP_FRAME_COUNT) // frame_skip + 1

    success, image = video.read()

    rotate_clockwise = False
    rotate_counterclockwise = False

    # If the video is vertical, prompt the user which way to rotate
    if image.shape[0] > image.shape[1]:
        # Display the image and wait for a right or left keypress
        plt.ion()
        plt.figure()
        plt.imshow(image)
        plt.show()
        desired = str(input("Press 'c' or 'w' to rotate clockwise or counter-clockwise."))
        plt.close()
        if desired == 'c':
            rotate_clockwise = True
            print('Rotating {} clockwise'.format(video_name))
        elif desired == 'w':
            rotate_counterclockwise = True
            print('Rotating {} counterclockwise'.format(video_name))
        else:
            print('Skipping {}'.format(video_name))
            return 0

    count = 0
    with tqdm(total=num_frames_to_process, leave=False) as pbar:
        while success:
            if count % frame_skip == 0:
                if rotate_clockwise:
                    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                if rotate_counterclockwise:
                    image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                cv2.imwrite(os.path.join(output_dir, '{}_frame_{}.jpg'.format(video_name.split('.')[0], count)), image)
                pbar.update(1)
            success, image = video.read()
            count += 1

    return num_frames_to_process


video_names = sorted(os.listdir(input_dir))
num_images = 0
for video_name in tqdm(iterable=video_names, total=len(video_names)):
    if video_name.split('.')[0] in already_processed_videos:
        print('Skipping {}'.format(video_name))
        continue
    num_added = extract_images(video_name)
    num_images += num_added

print('Extracted {} images from {} videos'.format(num_images, len(video_names)))
