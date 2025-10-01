import cv2
import numpy as np
from ultralytics import YOLO

def iou(boxa, boxb):
    xa = max(boxa[0], boxb[0])
    ya = max(boxa[1], boxb[1])
    xb = min(boxa[2], boxb[2])
    yb = min(boxa[3], boxb[3])

    interArea = max(0, xb - xa)*max(0, yb - ya)
    boxa_area = (boxa[2]-boxa[0]) * (boxa[3]-boxa[1])
    boxb_area = (boxb[2]-boxb[0]) * (boxb[3]-boxb[1])
    union = boxa_area + boxb_area - interArea

    return interArea / union if union > 0 else 0

def diou(boxa, boxb):
    iou_val = iou(boxa, boxb)

    xa_c = (boxa[0]+boxa[2]) / 2
    ya_c = (boxa[1]+boxa[3]) / 2
    xb_c = (boxb[0]+boxb[2]) / 2
    yb_c = (boxb[1]+boxb[3]) / 2

    center_dist = (xa_c - xb_c)**2 + (ya_c - yb_c)**2

    xc_min = min(boxa[0], boxb[0])
    yc_min = min(boxa[1], boxb[1])
    xc_max = max(boxa[2], boxb[2])
    yc_max = max(boxa[3], boxb[3])

    diag_len = (xc_max-xc_min)**2 + (yc_max-yc_min)**2

    return iou_val - (center_dist / diag_len if diag_len > 0 else 0)


