from models.experimental import attempt_load
import torch
from models.yolo import Detect

def add_nLM_to_pt(pathIn, pathOut, numberOfLandMark):
    ##### add number of landmark to pretrained .pt model
    dictPT = torch.load(pathIn)
    model = dictPT['model']
    layer_last = model.model[-1]
    assert isinstance(layer_last, Detect)

    if getattr(layer_last, "nLM", None) is None:
        layer_last.nLM = numberOfLandMark
    else:
        print("!!!!! already got, layer_last.nLM = ", layer_last.nLM)

    print(model.getLandMarkNum())
    print(dictPT['model'].getLandMarkNum())
    
    torch.save(dictPT, pathOut)

if __name__ == "__main__":
    add_nLM_to_pt(pathIn = "weights/Pretrained/yolov5n-0.5.pt", \
        pathOut = "weights/Pretrained/yolov5n-0.5-LM5.pt", numberOfLandMark = 106)