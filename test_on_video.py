import sys, os
sys.path.append(os.path.abspath(os.path.join('.', 'nanonets_object_tracking')))

# script argument imports
import argparse
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import cv2,pickle,sys

from deepsort import *

def get_parser():
    parser = argparse.ArgumentParser(description="SORT demo for pretrained checkpoints.")
    parser.add_argument(
        "--video-input",
        help="Path to video file."
    )
    parser.add_argument(
        "--input-mask-image",
        help="An optional mask for the given video, to focus on the sheeps."
    )
    parser.add_argument(
        "--model-checkpoint",
        default="nanonets_object_tracking/ckpts/model640.pt",
        help="Model checkpoint to use in deepsort"
        "If not specified, it will use nanonets_object_tracking/ckpts/model640.pt"
    )
    parser.add_argument(
        "--input-text-file",
        default="nanonets_object_tracking/det/det_demo.txt",
        help="A file from which to load bounding box detections made by detectron2. "
    )
    parser.add_argument(
        "--video-output",
        default="out",
        help="A file or directory to save output visualizations. "
        "Extension .avi will be appended to file name. "
    )
    parser.add_argument(
        "--draw-tracked-boxes",
        type=bool,
        default=True,
        help="Show final detections from SORT in output visualizations. "
    )
    parser.add_argument(
        "--draw-raw-detections",
        type=bool,
        default=False,
        help="Show original detections from detectron2 in output visualizations. "
    )

    return parser

def get_gt(image,frame_id,gt_dict):

    if frame_id not in gt_dict.keys() or gt_dict[frame_id]==[]:
        return None,None

    frame_info = gt_dict[frame_id]

    detections = []
    ids = []
    out_scores = []
    for i in range(len(frame_info)):

        coords = frame_info[i]['coords']

        x1,y1,w,h = coords
        x2 = x1 + w
        y2 = y1 + h

        xmin = min(x1,x2)
        xmax = max(x1,x2)
        ymin = min(y1,y2)
        ymax = max(y1,y2)   

        detections.append([x1,y1,w,h])
        out_scores.append(frame_info[i]['conf'])

    return detections,out_scores


def get_dict(filename):
    with open(filename) as f:   
        d = f.readlines()

    d = list(map(lambda x:x.strip(),d))

    last_frame = int(d[-1].split(',')[0])

    gt_dict = {x:[] for x in range(last_frame+1)}

    for i in range(len(d)):
        a = list(d[i].split(','))
        a = list(map(float,a))  

        coords = a[2:6]
        confidence = a[6]
        gt_dict[a[0]].append({'coords':coords,'conf':confidence})

    return gt_dict

def get_mask(filename):
    mask = cv2.imread(filename,0)
    mask = mask / 255.0
    return mask


if __name__ == '__main__':
    args = get_parser().parse_args()

    if args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

    
    #Load detections for the video. Options available: yolo,ssd and mask-rcnn
    assert os.path.isfile(args.input_text_file), "Please specify detections file."

    gt_dict = get_dict(args.input_text_file)
    #(1080,1920,3) (720,1280,3) 
    #video_dim = (1280,720)
    #video_dim = (1920,1080)
    video_dim = (width,height)

    #cap = cv2.VideoCapture('vdo.avi')
    cap = video

    # An optional mask for the given video, to focus on sheeps. 
    if args.input_mask_image:
        basename = os.path.basename(args.input_mask_image)
        basename = os.path.splitext(basename)[0]

        im = Image.open(args.input_mask_image)
        im = im.resize(video_dim, Image.ANTIALIAS)
        str_video_dim = str(video_dim)
        im.save(f"{basename}-{str_video_dim}.jpg", "JPEG")
        mask = get_mask(f"{basename}-{str_video_dim}.jpg")

        mask = np.expand_dims(mask,2)
        mask = np.repeat(mask,3,2)

    #Initialize deep sort.
    if args.model_checkpoint:
        assert os.path.isfile(args.model_checkpoint), "Please specify a modal checkpoint to use for deepsort"
        deepsort = deepsort_rbc(args.model_checkpoint)
    else:
        print("No checkpoint selected. Default checkpoint selecting... nanonets_object_tracking/ckpts/model640.pt")
        deepsort = deepsort_rbc()

    frame_id = 1


    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('ssd_out_3.avi',fourcc, 10.0, (1920,1080))
    # out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10,(720,1280))
    out = cv2.VideoWriter(
                filename=args.video_output + '.avi',
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"MJPG"), 
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
    while True:
        print(frame_id)     

        ret,frame = cap.read()
        if ret is False:
            frame_id+=1
            break

        if args.input_mask_image:
            frame = frame * mask
        frame = frame.astype(np.uint8)

        detections,out_scores = get_gt(frame,frame_id,gt_dict)

        if detections is None:
            print("No dets at seconds: ",int(frame_id*frames_per_second)) # we want to see a little before
            frame_id+=1
            continue

        detections = np.array(detections)
        out_scores = np.array(out_scores) 

        tracker,detections_class = deepsort.run_deep_sort(frame,out_scores,detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr() #Get the corrected/predicted bounding box
            id_num = str(track.track_id) #Get the ID for the particular track.
            features = track.features #Get the feature vector corresponding to the detection.

            #Draw bbox from tracker.
            if args.draw_tracked_boxes:
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                cv2.putText(frame, str(id_num),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

            #Draw bbox from detector. Just to compare.
            if args.draw_raw_detections:
                for det in detections_class:
                    bbox = det.to_tlbr()
                    cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,0), 2)

        #cv2.imshow('frame',frame)
        out.write(frame)

        frame_id+=1

    out.release()
