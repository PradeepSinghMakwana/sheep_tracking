#!/bin/bash
# Optional Requirements
# install dependencies: (use cu100 because colab is on CUDA 10.0)
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu100/index.html

git clone https://github.com/facebookresearch/detectron2.git

git clone https://github.com/abhyantrika/nanonets_object_tracking.git

wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-485FMDqf2J4RHwBgt41hHTvtiGcnbSA' -O video-clip.mp4

# Run frame-by-frame inference demo on this video (takes 3-4 minutes)
# Using a model trained on COCO dataset
# Use python3 on linux pc
python sheepFinder.py --config-file detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml --video-input video-clip.mp4 --confidence-threshold 0.6 \
  --opts MODEL.WEIGHTS detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl

# you can experiment with draw-raw-detections and draw-tracked-boxes in test_on_video.py to see results.
python test_on_video.py --video-input video-clip.mp4 --video-output vdo

# conversion if required
# sudo apt-get install ffmpeg
# ffmpeg -i vdo.mp4 -c:v libx264 -c:a libmp3lame -b:a 384K vdo.avi

# Please see some included file code to understand new ways of interacting with this file.
