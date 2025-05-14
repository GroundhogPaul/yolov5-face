"""
before test the onnx model on OpenCV-C++, first test on OpenCV-Python
"""

import cv2
import numpy as np
import copy
from detect_face import show_results
from utils.general import xywh2xyxy

WinNN, HinNN = 128, 96
imgInBGR = cv2.imread("./runs/AD_128_96.bmp")
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
    size=(HinNN, WinNN),
#     # mean=(0.485, 0.456, 0.406),  # ImageNet 均值
#     # std=(0.229, 0.224, 0.225),    # ImageNet 标准差
    swapRB=True,
#     # crop=False
)
imgToPlot = blob[0].transpose(1,2,0)
# cv2.imshow("", imgToPlot)

# ------ NNet work load and input ----- # 
# net = cv2.dnn.readNetFromONNX("weights/Pretrained/yolov5n-0.5.onnx")
net = cv2.dnn.readNetFromONNX("weights/Pretrained/best.onnx")
net.setInput(blob)
output = net.forward()

print(type(output))

# ----- filt the output according to the objectness ----- #
output = output[0] # batch 0
output = output[output[:,4] > 0.5] # obj > thre

imgOutBGR_pad = copy.deepcopy(imgInBGR_pad) 
nLM = 5
for face in output:
    xywh = face[:4].tolist()
    xyxy = np.copy(xywh)
    xywh[0] = xyxy[0] - xyxy[2] / 2  # top left x
    xywh[1] = xyxy[1] - xyxy[3] / 2  # top left y
    xywh[2] = xyxy[0] + xyxy[2] / 2  # bottom right x
    xywh[3] = xyxy[1] + xyxy[3] / 2  # bottom right y
    conf = face[4]
    landmarks = face[5:5+nLM*2].tolist()
    class_num = face[5+nLM*2]

    imgPlot = show_results(imgInBGR_pad, imgOutBGR_pad, xywh, conf, landmarks, class_num, nLM = nLM)
cv2.imwrite("testONNX.jpg", imgPlot)