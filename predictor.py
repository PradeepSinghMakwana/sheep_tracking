# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# Bring your packages onto the path
import sys, os
sys.path.append(os.path.abspath(os.path.join('../detectron2', 'detectron2')))

import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch
import numpy as np

import detectron2

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode


class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def _frame_from_video(self, video):
        frame_id = 0
        while video.isOpened():
            success, frame = video.read()
            frame_id+=1
            if success:
                yield (frame_id,frame)
            else:
                break

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: bounding boxes in video frame in the format required by SORT.
        """
        print('Processing video...This may take few minutes.')

        def process_predictions(frame_info,predictions):
            frame_id,frame = frame_info
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            if "instances" in predictions:
                pred_classes = np.array(predictions["instances"].pred_classes.to(self.cpu_device))
                frame_id = np.ones(len(pred_classes),dtype='int')*frame_id # broadcast frame identifier
                negative_one = np.ones(len(pred_classes),dtype='int')*-1 
                boxes = np.array([np.array(x) for x in predictions["instances"].pred_boxes.to(self.cpu_device)])
                if boxes.ndim==2:
                    boxes[:,2] = boxes[:,2] - boxes[:,0] # width = x2 - x1
                    boxes[:,3] = boxes[:,3] - boxes[:,1] # height = y2 - y1
                elif boxes.ndim==1 and len(boxes)>0: # only if there exists a box
                    boxes[2] = boxes[2] - boxes[0] # width = x2 - x1
                    boxes[3] = boxes[3] - boxes[1] # height = y2 - y1
                scores = np.array(predictions["instances"].scores.to(self.cpu_device))
                preds = np.c_[pred_classes,frame_id,negative_one,boxes,scores,negative_one,negative_one,negative_one]
                vis_frame = []
                for pred in preds:
                    if pred[0]==18: # pass predictions for sheeps only
                        vis_frame.append(pred[1:])

            return np.array(vis_frame)

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt,(frame_id, frame) in enumerate(frame_gen):
                frame_data.append((frame_id, frame))
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame_id,frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions((frame_id,frame),prediction)

            while len(frame_data):
                frame_id,frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions((frame_id,frame),prediction)
        else:
            for frame_id,frame in frame_gen:
                yield process_predictions((frame_id,frame),self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because detecting and recognising may take considerably amount of time,
    this helps improve throughput when working with videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
