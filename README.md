# sheep_tracking
This repo uses [detectron2](https://github.com/facebookresearch/detectron2) and deepsort implementation [nanonets_object_tracking](https://github.com/abhyantrika/nanonets_object_tracking) to track moving sheep in a video.

## Setup environment
The project results can be produced by running the below code in a GPU environment in a google colab notebook.

1. Type "google colab" on google search bar.
2. Open google colab and create a new notebook.  
3. Go to Edit -> Notebook settings, select GPU as Hardware accelerator, and click Save.  
4. Copy paste the below code in the first cell, and click play.
  
```
!git clone https://github.com/PradeepSinghMakwana/sheep_tracking.git
!mv sheep_tracking/* ./
!rm -r sheep_tracking
```
  
## Setup dependencies and files
  
5. Add another code cell by clicking ` + Code` button in the top left corner of the screen. 
6. Copy paste the below code in the second cell, and click play.
```
# install dependencies: (use cu100 because colab is on CUDA 10.0)
!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu100/index.html

!git clone https://github.com/facebookresearch/detectron2.git

!git clone https://github.com/abhyantrika/nanonets_object_tracking.git

""" get the input video file """
!wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=1-485FMDqf2J4RHwBgt41hHTvtiGcnbSA" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-485FMDqf2J4RHwBgt41hHTvtiGcnbSA" -O video-clip.mp4 && rm -rf /tmp/cookies.txt
```

## Run
  
7. Copy paste the below code in the third cell, and click play.
  
```
!python sheepFinder.py --config-file detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml --video-input video-clip.mp4 --confidence-threshold 0.6 \
  --opts MODEL.WEIGHTS detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl

!python test_on_video.py --video-input video-clip.mp4 --video-output vdo
```
  
  
## Output
  
If everything works fine you will see a `vdo.avi` file created. You can download it, by right click on it, select download.
  
  
## Run on your video
  
To run the tracker on your video. Follow steps from 1 to 6 and then:  
  
From left hand side select files and click Upload, and upload a mp4 video of your choice in google colab. **Please upload only .mp4 videos as input video file.**  
  
Once your video file is uploaded, you can rename it to video-clip.mp4. Follow step 7 to generate tracking results on your video.  
  
  
## Possible modification
  
If you want only a single, but, some other class to be detected, you can change it in `predictor.py` file. For this you need to download the current predictor.py file and upload your changed file in its place. Happy coding!!
  
  
## For learners and enthusiasts
  
Above, you may have executed the following lines. You can change the config-file attribute to change the detection modal. You can experiment with some other parameters given in the source code of files `sheepFinder.py` and `test_on_video.py`.  You can find the other config models in [detectron2 MODEL ZOO](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md). You can find how to change this parameter and some other examples in the colab notebook associated with detectron2 [Here](https://github.com/facebookresearch/detectron2).
  
```
!python sheepFinder.py --config-file detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml --video-input video-clip.mp4 --confidence-threshold 0.6 \
  --opts MODEL.WEIGHTS detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl

""" you can experiment with draw-raw-detections and draw-tracked-boxes in test_on_video.py to see results. """
!python test_on_video.py --video-input video-clip.mp4 --video-output vdo
```  
  
For it to work best, the deep sort model used in `test_on_video.py` needs training on moving sheep data. I am just using a model trained on moving humans just like MOT challenge.
  
## Finally  
  
This project is completely my creation (although, I have used some demo files). It's here you can use it for free. For any questions or suggestion, feel free to create an issue.
  
Original Author: Pradeep Singh Makwana (pradeepsinghmakwana@gmail.com)
