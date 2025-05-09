# -*- coding: UTF-8 -*-
import argparse
import time
from pathlib import Path
import sys
import os

import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import copy
from models.yolo import Detect

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.experimental import attempt_load
from utils.datasets import letterbox, img_formats, vid_formats, LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression_face, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


def load_model(weights, device):
    model = attempt_load(weights, map_location=device)  # load FP32 model
    return model


def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        # wh pad is half on both side, correspond to datasets.py::letterbox()
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    assert coords.shape[-1] % 2 == 0, f"coords.shape[-1] should be even, because it's pairs of x,y, current = {coords.shape[-1]}."

    lenLandMarkXY = coords.shape[-1]
    lstEven = [i for i in range(0, lenLandMarkXY, 2)] # indices of x
    lstOdd = [i for i in range(1, lenLandMarkXY, 2)] # indices of y
    
    coords[:, lstEven] -= pad[0]  # x padding
    coords[:, lstOdd] -= pad[1]  # y padding
    coords[:, :lenLandMarkXY] /= gain
    #clip_coords(coords, img0_shape)
    for i in lstEven:
        coords[:, i].clamp_(0, img0_shape[1])  # x1
        coords[:, i+1].clamp_(0, img0_shape[0])  # y1

    # coords[:, 0].clamp_(0, img0_shape[1])  # x1
    # coords[:, 1].clamp_(0, img0_shape[0])  # y1
    # coords[:, 2].clamp_(0, img0_shape[1])  # x2
    # coords[:, 3].clamp_(0, img0_shape[0])  # y2
    # coords[:, 4].clamp_(0, img0_shape[1])  # x3
    # coords[:, 5].clamp_(0, img0_shape[0])  # y3
    # coords[:, 6].clamp_(0, img0_shape[1])  # x4
    # coords[:, 7].clamp_(0, img0_shape[0])  # y4
    # coords[:, 8].clamp_(0, img0_shape[1])  # x5
    # coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def show_results(img, xyxy, conf, landmarks, class_num):
    h,w,c = img.shape
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0])
    y1 = int(xyxy[1])
    x2 = int(xyxy[2])
    y2 = int(xyxy[3])
    img = img.copy()
    
    cv2.rectangle(img, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    for i in range(5):
        point_x = int(landmarks[2 * i])
        point_y = int(landmarks[2 * i + 1])
        cv2.circle(img, (point_x, point_y), tl+1, clors[i], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


def detect(
    model,
    source,
    device,
    project,
    name,
    exist_ok,
    save_img,
    view_img
):
    # Load model
    img_size = 640
    conf_thres = 0.6
    iou_thres = 0.5
    imgsz=(640, 640)
    
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    Path(save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    is_file = Path(source).suffix[1:] in (img_formats + vid_formats)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    
    # Dataloader
    if webcam:
        print('loading streams:', source)
        dataset = LoadStreams(source, img_size=imgsz)
        bs = 1  # batch_size
    else:
        print('loading images', source)
        dataset = LoadImages(source, img_size=imgsz)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    for path, imRGBltd_ChHW, im0s, vid_cap in dataset: # ltd: lettered
        
        if len(imRGBltd_ChHW.shape) == 4:
            orgimg = np.squeeze(imRGBltd_ChHW.transpose(0, 2, 3, 1), axis= 0)
        else:
            imRGBltd_HWCh = imRGBltd_ChHW.transpose(1, 2, 0)
        h0, w0 = imRGBltd_HWCh.shape[:2]  # orig hw
        
        imBGRltd_HWCh = cv2.cvtColor(imRGBltd_HWCh, cv2.COLOR_BGR2RGB)
        # img0BGR = copy.deepcopy(imBGRltd_HWCh)
        # r = img_size / max(h0, w0)  # resize image to img_size
        # if r != 1:  # always resize down, only resize up if training with augmentation
        #     interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        #     img0BGR = cv2.resize(imgBGRltd_HWCh, (int(w0 * r), int(h0 * r)), interpolation=interp)

        imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

        imgBGRltd_HWCh = letterbox(imBGRltd_HWCh, new_shape=imgsz)[0]
        # cv2.imshow("", imgBGR)
        # cv2.waitKey(0)
        # Convert from w,h,c to c,w,h
        imgBGRltd_ChHW = imgBGRltd_HWCh.transpose(2, 0, 1).copy()

        imgBGRltd_ChHW = torch.from_numpy(imgBGRltd_ChHW).to(device)
        imgBGRltd_ChHW = imgBGRltd_ChHW.float()  # uint8 to fp16/32
        imgBGRltd_ChHW /= 255.0  # 0 - 255 to 0.0 - 1.0
        if imgBGRltd_ChHW.ndimension() == 3:
            imgBGRltd_ChHW = imgBGRltd_ChHW.unsqueeze(0)

        # Inference
        pred = model(imgBGRltd_ChHW)[0]
        
        # Apply NMS
        pred = non_max_suppression_face(pred, conf_thres, iou_thres, nLM = model.getLandMarkNum())
        print(len(pred[0]), 'face' if len(pred[0]) == 1 else 'faces')

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            
            p = Path(p)  # to Path
            save_path = str(Path(save_dir) / p.name)  # im.jpg

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(imgBGRltd_ChHW.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                det[:, 5:15] = scale_coords_landmarks(imgBGRltd_ChHW.shape[2:], det[:, 5:15], im0.shape).round()

                for j in range(det.size()[0]):
                    xyxy = det[j, :4].view(-1).tolist()
                    conf = det[j, 4].cpu().numpy()
                    landmarks = det[j, 5:15].view(-1).tolist()
                    class_num = det[j, 15].cpu().numpy()
                    
                    im0 = show_results(im0, xyxy, conf, landmarks, class_num)
            
            if view_img:
                cv2.imshow('result', im0)
                k = cv2.waitKey(1)
                    
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    save_path = "./detect_face_out.jpg"
                    cv2.imwrite(save_path, im0)
                    print(os.path.basename(__file__), ": save output image to ", save_path)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    try:
                        vid_writer[i].write(im0)
                    except Exception as e:
                        print(e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/exp5/weights/last.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save-img', action='store_true', help='save results')
    parser.add_argument('--view-img', action='store_true', help='show results')
    opt = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(opt.weights, device)
    detect(model, opt.source, device, opt.project, opt.name, opt.exist_ok, opt.save_img, opt.view_img)
