"""
before test the onnx model on OpenCV-C++, first test on OpenCV-Python
"""

import cv2
import numpy as np
import copy
import sys
import os
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from detect_face import show_results, export_cat_parse, non_max_suppression_face
from utils.general import xywh2xyxy

WinNN, HinNN = 640, 512
imgInBGR = cv2.imread("./runs/AD_640_480.bmp")
assert imgInBGR.shape[0] <= HinNN
assert imgInBGR.shape[1] <= WinNN

# ----- pad input image ----- #
H0, W0, CH = imgInBGR.shape
delta_w = WinNN - W0
delta_h = HinNN - H0
 
top = delta_h // 2
bottom = delta_h - top
left = delta_w // 2
right = delta_w - left

imgInBGR_pad = cv2.copyMakeBorder(
    imgInBGR, 
    top, bottom, left, right, 
    cv2.BORDER_CONSTANT, 
    value=[128, 128, 128]
)
# cv2.imwrite("ONNXinput.jpg", imgInBGR_pad)

# ----- normalize and to RGB ----- #
blob = cv2.dnn.blobFromImage(
    imgInBGR_pad,
    scalefactor=1.0/255,
    size=(WinNN, HinNN),
#     # mean=(0.485, 0.456, 0.406),  # ImageNet 均值
#     # std=(0.229, 0.224, 0.225),    # ImageNet 标准差
    swapRB=True,
#     # crop=False
)

# ------ NNet work load and input ----- # 
# net = cv2.dnn.readNetFromONNX("weights/Pretrained/yolov5n-0.5.onnx")
net = cv2.dnn.readNetFromONNX("weights/Pretrained/yolov5n-0.5-face128lm_20250514_640x640.onnx")
net.setInput(blob)
print("Onnx: input type: ", type(blob), ", input shape:", blob.shape)

# ----------- NNet work load and input ---------- # 
output = net.forward()

# ----- filt the output according to the objectness ----- #
print("Onnx: output type: ", type(output), ", output shape:", output.shape)
pred = output.squeeze()
nAcLayer = 3
nAcPerLayer = 3
strides = np.array([8, 16, 32])
anchor_grid =  np.array([[[4,5],  [8,10],  [13,16]],  # P3/8
                        [[23,29],  [43,55],  [73,105]], # P4/16
                        [[146,217],  [231,300],  [335,433]]])  # P5/32

pred = torch.from_numpy(pred)
output =  export_cat_parse(pred, HinNN, WinNN, nAcLayer, nAcPerLayer, strides, anchor_grid, conf_thres = 0.5)
output = output[0]
output = output[output[:,4] > 0.5] # obj > thre

# ----- NMS -----
nLM = 106
output = output[None, :]
output = non_max_suppression_face(output, nLM = nLM)
output = output[0]

imgOutBGR_pad = copy.deepcopy(imgInBGR_pad) 
for face in output:
    xywh = face[:4].tolist()
    # xyxy = np.copy(xywh)
    # xywh[0] = xyxy[0] - xyxy[2] / 2  # top left x
    # xywh[1] = xyxy[1] - xyxy[3] / 2  # top left y
    # xywh[2] = xyxy[0] + xyxy[2] / 2  # bottom right x
    # xywh[3] = xyxy[1] + xyxy[3] / 2  # bottom right y
    conf = face[4].item()
    landmarks = face[5:5+nLM*2].tolist()
    class_num = face[5+nLM*2]

    imgPlot = show_results(imgInBGR_pad, imgOutBGR_pad, xywh, conf, landmarks, class_num, nLM = nLM)
cv2.imwrite("testONNX.jpg", imgPlot)