# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', 'detectron2')))

import numpy as np
import argparse
import multiprocessing as mp
import os
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--output-text-file",
        default="nanonets_object_tracking/det/det_demo.txt",
        help="A file or directory to save output bounding box results. "
        "This file will be used by sort as input detections. ",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--parallel",
        type=bool,
        default=False,
        help="Whether to use parrallel capabilities",
    )
    parallel=False
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg,parallel=args.parallel)
    assert os.path.isfile(args.video_input), "Please specify a video file with --video-input"
    if args.video_input:
        video = cv2.VideoCapture(args.video_input)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = os.path.basename(args.video_input)

        if os.path.isfile(args.output_text_file):
            print("Output text file already exist, replacing it.") # Does bob.txt exist?  Is it a file, or a directory?
        elif os.path.isdir(args.output_text_file):
            print("Saving output in: ", os.path.join(args.output_text_file,'det_demo.txt'))

        # assert (os.path.isfile(args.output_text_file) or os.path.isdir(args.output_text_file)), "Please specify --output-text-file"
        if args.output_text_file:
            output_text_file = np.array([])
            output_is_empty = True
        for vis_frame in tqdm.tqdm(demo.run_on_video(video), total=num_frames):
            if args.output_text_file:
                if len(vis_frame)==0:
                    continue
                if vis_frame.ndim==1:
                    vis_frame = np.array([vis_frame]) # wrap into a new axis
                if output_is_empty:
                   output_text_file = vis_frame
                   output_is_empty = False
                else:
                   output_text_file = np.concatenate([output_text_file,vis_frame])
        # release input video file
        video.release()
        if args.output_text_file:
            np.savetxt(args.output_text_file, output_text_file, delimiter=",", fmt=['%d','%d','%.3f','%.3f','%.3f','%.3f','%.3f','%d','%d','%d'])
