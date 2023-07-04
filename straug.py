import json
from PIL import Image
from straug.warp import Curve
import os
import argparse
import os
import numpy as np
from PIL import Image
import cv2

from straug.warp import Curve, Distort, Stretch
from straug.geometry import Rotate, Perspective, Shrink, TranslateX, TranslateY
from straug.blur import GaussianBlur, DefocusBlur, MotionBlur, GlassBlur, ZoomBlur
from straug.camera import Contrast, Brightness, JpegCompression, Pixelate
from straug.noise import GaussianNoise, ShotNoise, ImpulseNoise, SpeckleNoise
from straug.pattern import VGrid, HGrid, Grid, RectGrid, EllipseGrid
from straug.process import Posterize, Solarize, Invert, Equalize, AutoContrast, Sharpness, Color
from straug.weather import Fog, Snow, Frost, Rain, Shadow
class Random_StrAug(object):
    def __init__(self, using_aug_types, prob_list = None):
        self.aug_list = []
        if 'warp' in using_aug_types :
            self.aug_list.append([Curve(), Distort(), Stretch()]) 
        if 'geometry' in using_aug_types :
            self.aug_list.append([Rotate(), Perspective(), Shrink(), TranslateX(), TranslateY()]) 
        if 'blur' in using_aug_types :
            self.aug_list.append([GaussianBlur(), DefocusBlur(), MotionBlur(), GlassBlur(), ZoomBlur()]) 
        if 'noise' in using_aug_types :
            self.aug_list.append([GaussianNoise(), ShotNoise(), ImpulseNoise(), SpeckleNoise()]) 
        if 'camera' in using_aug_types :
            self.aug_list.append([Contrast(), Brightness(), JpegCompression(), Pixelate()]) 
        if 'pattern' in using_aug_types :
            self.aug_list.append([VGrid(), HGrid(), Grid(), RectGrid(), EllipseGrid()]) 
        if 'process' in using_aug_types :
            self.aug_list.append([Posterize(), Solarize(), Invert(), Equalize(), AutoContrast(), Sharpness(), Color()]) 
        if 'weather' in using_aug_types :
            self.aug_list.append([Fog(), Snow(), Frost(), Rain(), Shadow()]) 
    
        self.mag_range = np.random.randint(-1, 3)
        if prob_list is None :
            self.prob_list = [0.5] * len(self.aug_list)
        else:
            assert len(self.aug_list) == len(prob_list), "The length of 'prob_list' must be the same as the number of augmentations used."
            self.prob_list = prob_list

    def __call__(self, img):
        for i in range(len(self.aug_list)):
            img = self.aug_list[i][np.random.randint(0, len(self.aug_list[i]))](img, mag = self.mag_range, prob = self.prob_list[i])

        return img
# Define augmenter function
random_StrAug = Random_StrAug(using_aug_types = ['warp', 'geometry', 'blur', 'noise', 'camera', 'pattern', 'process', 'weather'],
                                  prob_list = [0.5, 0.3, 0.3, 0.2, 0.2, 0.1, 0.1, 0.1])
# Load annotation file
with open("/mmocr/mmocr/data/icdar2015/textdet_train.json") as f:
    ann = json.load(f)

# Define output directory

# Loop over each image in the dataset
for data in ann["data_list"]:
    img_path = "/mmocr/mmocr/data/icdar2015/" + data["img_path"]
    img = Image.open(img_path)

    # Loop over each text instance in the image
    for instance in data["instances"]:
        bbox = instance["bbox"]
        x1, y1, x2, y2 = map(int, bbox)
        text_region = img.crop((x1, y1, x2, y2))

        # Apply elastic deformation to text region
        aug_text = random_StrAug(text_region)

        # Paste augmented text region back into original image
        img.paste(aug_text, (x1, y1, x2, y2))

    # Save augmented image to file
    file_name = os.path.basename(data["img_path"])
    output_path = os.path.join("/mmocr/mmocr/data/icdar2015/textdet_imgs/train", "augmented_" + file_name)
    img.save(output_path)
