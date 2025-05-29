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

    return coords

def export_cat_parse(pred: torch.Tensor, H, W, nAcLayer, nAcPerLayer, strides, anchor_grid, conf_thres):
    assert isinstance(pred, torch.Tensor), "pred must be a torch.Tensor" # to use sigmoid
    assert len(pred.shape) == 2, "shape of pred should be like (20160, 218) or (20160, 16)"
    conf_thres_inv_sigmoid = np.log(conf_thres / (1 - conf_thres))
    boxOffset = 0
    for iAcLayer in range(nAcLayer):
        detRatio = strides[iAcLayer]
        assert H % detRatio == 0
        assert W % detRatio == 0
        HAcLayerOut, WAcLayerOut = H // detRatio, W // detRatio

        outAcLayer = pred[boxOffset: boxOffset + HAcLayerOut * WAcLayerOut * nAcPerLayer, :]

        nLM = 106
        nc = 1
        for iAc in range(nAcPerLayer):
            outAcLayerIac = outAcLayer[iAc * HAcLayerOut * WAcLayerOut : (iAc+1) * HAcLayerOut * WAcLayerOut, :] # current Anchor Layer, iAc anchor box
            for hthAcBox in range(HAcLayerOut):
                for wthAcBox in range(WAcLayerOut):
                    face = outAcLayerIac[(hthAcBox*WAcLayerOut + wthAcBox), :]
                    # print(f"L:{nAcLayer}, Ac:{nAcPerLayer}, h:{hthAcBox}, w:{wthAcBox}, ", "{:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}".format(face[0], face[1], face[2], face[3], face[4]))
                    if face[4] < conf_thres_inv_sigmoid:
                        continue
                    class_range = list(range(5)) + list(range(5+2*nLM,5+2*nLM+nc))
                    face[class_range] = face[class_range].sigmoid()

                    face[0:2] = (face[0:2] * 2. - 0.5 + np.array([wthAcBox, hthAcBox])) * strides[iAcLayer]  # xy
                    face[2:4] = (face[2:4] * 2) ** 2 * anchor_grid[iAcLayer, iAc] # wh

                    for iLM in range(nLM):
                        face[..., 5+2*iLM: 7+2*iLM] = face[..., 5+2*iLM: 7+2*iLM] * anchor_grid[iAcLayer, iAc] + np.array([wthAcBox, hthAcBox]) * strides[iAcLayer] # landmark x1, y1
                        
        boxOffset += HAcLayerOut * WAcLayerOut * nAcPerLayer
        # print(HAcLayerOut, WAcLayerOut)
    pred = pred[None , :]
    return pred

def export_cat_parse_detectLayer(pred, H, W, detectLayer: Detect, conf_thres):
    layerDetect = model.model[-1]
    nAcLayer = layerDetect.nl
    nAcPerLayer = layerDetect.na
    strides = layerDetect.stride.to(torch.int32).numpy()
    anchor_grid = layerDetect.anchor_grid.numpy().squeeze()

    pred = export_cat_parse(pred, H, W, nAcLayer, nAcPerLayer, strides, anchor_grid, conf_thres)
    return pred

def show_results(imgIn, imgOut, xyxy, conf, landmarks, class_num, nLM = 5):
    h,w,c = imgIn.shape
    # ----- scale up to 640 ----- # or the image would be too small to observe the landmarks
    rScaleUp = int(max(640 / max(h,w), 1)) # ratio scale up
    h, w = h * rScaleUp, w* rScaleUp
    rScaleUp = 1
    if rScaleUp == 1 or max(imgOut.shape) == 640:
        pass
    else:
        imgOut = cv2.resize(imgIn, (w, h), interpolation=cv2.INTER_LINEAR)
    
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    x1 = int(xyxy[0] * rScaleUp)
    y1 = int(xyxy[1] * rScaleUp)
    x2 = int(xyxy[2] * rScaleUp)
    y2 = int(xyxy[3] * rScaleUp)
    
    cv2.rectangle(imgOut, (x1,y1), (x2, y2), (0,255,0), thickness=tl, lineType=cv2.LINE_AA)

    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]

    for i in range(nLM):
        point_x = int(landmarks[2 * i] * rScaleUp)
        point_y = int(landmarks[2 * i + 1] * rScaleUp)
        cv2.circle(imgOut, (point_x, point_y), tl, clors[i%5], -1)

    tf = max(tl - 1, 1)  # font thickness
    label = str(conf)[:5]
    cv2.putText(imgOut, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return imgOut


def detect(
    model,
    source,
    device,
    project,
    name,
    exist_ok,
    save_img,
    view_img, 
    export_cat,
    img_size
):
    # Load model
    conf_thres = 0.5
    iou_thres = 0.25
    
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    Path(save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    is_file = Path(source).suffix[1:] in (img_formats + vid_formats)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    
    # Dataloader
    if webcam:
        print('loading streams:', source)
        dataset = LoadStreams(source, img_size=img_size)
        bs = 1  # batch_size
    else:
        print('loading images', source)
        dataset = LoadImages(source, img_size=img_size)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    for path, imgRGBltd_ChHW, im0s, vid_cap in dataset: # ltd: lettered
        
        if len(imgRGBltd_ChHW.shape) == 4:
            orgimg = np.squeeze(imgRGBltd_ChHW.transpose(0, 2, 3, 1), axis= 0)
        else:
            imgRGBltd_HWCh = imgRGBltd_ChHW.transpose(1, 2, 0)
        h0, w0 = imgRGBltd_HWCh.shape[:2]  # orig hw
        
        imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size

        imgRGBltd_HWCh = letterbox(imgRGBltd_HWCh, new_shape=imgsz)[0]
        print("image input to NN has size = ", imgRGBltd_HWCh.shape)
        # cv2.imshow("", imgRGBltd_HWCh)
        # cv2.waitKey(0)
        # Convert from w,h,c to c,w,h
        imgRGBltd_ChHW = imgRGBltd_HWCh.transpose(2, 0, 1).copy()

        imgRGBltd_ChHW = torch.from_numpy(imgRGBltd_ChHW).to(device)
        imgRGBltd_ChHW = imgRGBltd_ChHW.float()  # uint8 to fp16/32
        imgRGBltd_ChHW /= 255.0  # 0 - 255 to 0.0 - 1.0
        if imgRGBltd_ChHW.ndimension() == 3:
            imgRGBltd_ChHW = imgRGBltd_ChHW.unsqueeze(0)

        # Inference
        model.model[-1].export_cat = export_cat
        print("Pt: input type: ", type(imgRGBltd_ChHW), ", input shape:", imgRGBltd_ChHW.shape)
        pred = model(imgRGBltd_ChHW)[0]
        print("Pt: output type: ", type(pred), ", output shape:", pred.shape)

        if export_cat:
            layerDetect = model.model[-1]
            pred = export_cat_parse_detectLayer(pred, imgRGBltd_ChHW.shape[2], imgRGBltd_ChHW.shape[3], layerDetect, conf_thres)
        
        # Apply NMS
        conf_thres = 0.01
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

            imPlot = copy.deepcopy(im0) 
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(imgRGBltd_ChHW.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class

                nLM = 106 # TODO magic number
                det[:, 5:5+nLM*2] = scale_coords_landmarks(imgRGBltd_ChHW.shape[2:], det[:, 5:5+nLM*2], im0.shape).round()

                for j in range(det.size()[0]):
                    xyxy = det[j, :4].view(-1).tolist()
                    conf = det[j, 4].cpu().numpy()
                    landmarks = det[j, 5:5+nLM*2].view(-1).tolist()
                    class_num = det[j, 5+nLM*2].cpu().numpy()

                    imPlot = show_results(im0, imPlot, xyxy, conf, landmarks, class_num, nLM = nLM)
            
            if view_img:
                cv2.imshow('result', imPlot)
                k = cv2.waitKey(1)
                    
            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    save_path = "./detFaceOut.jpg"
                    cv2.imwrite(save_path, imPlot)
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
    parser.add_argument('--img_size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--save-img', action='store_true', help='save results')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--export_cat', type=bool, default=False, help='the last Layer of NN only cat the output for C++ mode')
    opt = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(opt.weights, device)
    detect(model, opt.source, device, opt.project, opt.name, opt.exist_ok, opt.save_img, opt.view_img, opt.export_cat, opt.img_size)
