# K2020-KLTN-Scene-Text-Detection-Using-A-Segmentation-Method
# OCR Code Execution Guide

This guide will walk you through the steps required to implement and run the STD (Scene Text Detection) code using the mmocr library. Please follow the instructions below to set up the environment, install the necessary dependencies, and execute the code.

## Environment Setup

```shell
# Install openmim
!pip install -U openmim

# Install mmengine
!mim install mmengine

# Install mmdet (version 3.0.0rc6)
!mim install mmdet==3.0.0rc6

# Clone the mmocr repository
!git clone https://github.com/Banhkun/mmocr.git
%cd mmocr

# Install dependencies
!pip install -r requirements.txt

# Install mmocr
!pip install -v -e .
```
## Dataset Setup
```shell
# Change directory to mmocr
%cd /content/mmocr/

# Prepare the dataset for text detection task (e.g., icdar2015)
!python /content/mmocr/tools/dataset_converters/prepare_dataset.py icdar2015 --task textdet
```
## Additional Configuration
```shell
# Change directory to mmocr
%cd /content/mmocr/

# Install mmdet (version 3.0.0rc6)
!mim install mmdet==3.0.0rc6

# Copy necessary files
!cp -r /content/mmocr/mask_target.py /usr/local/lib/python3.9/dist-packages/mmdet/structures/mask/
!cp /content/mmocr/cascade_roi_head_tool_mmocr.py /usr/local/lib/python3.9/dist-packages/mmdet/models/roi_heads/cascade_roi_head.py
```
## Training
```shell
# Change directory to mmocr
%cd /content/mmocr/

# Run the training script with the specified configuration file
!python tools/train.py /content/drive/MyDrive/OCR/ckpt/cascade-diffusion-icdar15/cascade_diffusion.py --resume
```
## Make sure to replace the file paths and other arguments as per your setup.