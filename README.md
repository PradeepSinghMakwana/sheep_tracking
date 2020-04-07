# sheep_tracking
This repo uses [detectron2](https://github.com/facebookresearch/detectron2) and deepsort implementation [nanonets_objectect_tracking](https://github.com/abhyantrika/nanonets_object_tracking) to track moving sheep in a video.

## setup environment
The project results can be produced by running the below code in a GPU environment in a google colab notebook.

1. First download the repo in a zip.  
2. Open google colab and create a new notebook.  
3. Go to Edit -> Notebook settings, select GPU as Hardware accelerator, and click Save.  
4. From left hand side select files and click Upload, and upload the downloaded zip in google colab.  
5. Copy paste the below code in the first cell, and click play.

```
import zipfile
with zipfile.ZipFile('sheep_tracking.zip', 'r') as zip_ref:
  zip_ref.extractall('/content')
!mv sheep_tracking/* ./
!rm sheep_tracking.zip
!rmdir sheep_tracking
```

## run
6. Copy paste the below code in the second cell, and click play.

```
!chmod u+x ./setup_N_run.sh
!./setup_N_run.sh
```

## output
If everything works fine you will see a `vdo.avi` file created. You can download it, by right click on it, select download.

## possible modifications
If you want some other class to be detected you can just change it in predictor.py file. Happy coding. 
  
  
  
## for experts
The setup_N_run.sh contains the following lines. You can change the config-file attribute to change the detection modal. You can experiment with some other parameters given in the source code of files `sheepFinder.py` and `predictor.py`.

These lines are from setup_N_run.sh
```
!python sheepFinder.py --config-file detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml --video-input video-clip.mp4 --confidence-threshold 0.6 \
  --opts MODEL.WEIGHTS detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl

""" you can experiment with draw-raw-detections and draw-tracked-boxes in test_on_video.py to see results. """
!python test_on_video.py --video-input video-clip.mp4 --video-output vdo
```

This project is completely my creation (although, I have used some demo files) which wasn't paid by one of my freelancer.com clients. But it's needed by many good people.  So, I make it public. It's here you can use it for free. For any questions, just create an issue.

Original Author: Pradeep Singh Makwana (pradeepsinghmakwana@gmail.com)
