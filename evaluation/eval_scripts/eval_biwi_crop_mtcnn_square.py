import argparse
import logging
import os

import cv2
import numpy as np

import eval_biwi_crop as base


def square_crop_from_box(img_bgr, box_xyxy, crop_size=256):
    """
    Build a square crop centered on the detector box, padding with zeros if needed,
    then resize to the requested crop size.
    """
    img_h, img_w = img_bgr.shape[:2]
    x1, x2, y1, y2 = [int(v) for v in box_xyxy]

    box_w = max(1, x2 - x1 + 1)
    box_h = max(1, y2 - y1 + 1)
    side = max(box_w, box_h)

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    sq_x1 = int(round(cx - side / 2))
    sq_y1 = int(round(cy - side / 2))
    sq_x2 = sq_x1 + side
    sq_y2 = sq_y1 + side

    src_x1 = max(0, sq_x1)
    src_y1 = max(0, sq_y1)
    src_x2 = min(img_w, sq_x2)
    src_y2 = min(img_h, sq_y2)

    canvas = np.zeros((side, side, 3), dtype=img_bgr.dtype)
    dst_x1 = src_x1 - sq_x1
    dst_y1 = src_y1 - sq_y1
    dst_x2 = dst_x1 + (src_x2 - src_x1)
    dst_y2 = dst_y1 + (src_y2 - src_y1)

    canvas[dst_y1:dst_y2, dst_x1:dst_x2] = img_bgr[src_y1:src_y2, src_x1:src_x2]
    return cv2.resize(canvas, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)


def crop_face_mtcnn_square_for_vggt(img_bgr, detector, prev_box, crop_size=256, ad=0.4):
    """
    Reuse the same detector/tracker logic as the shared MTCNN evaluator, but convert the
    selected box into a square crop before resize so VGGT does not get aspect-ratio distortion.
    """
    _, new_box = base.crop_face_mtcnn(
        img_bgr,
        detector,
        prev_box,
        crop_size=crop_size,
        ad=ad,
    )
    crop_bgr = square_crop_from_box(img_bgr, new_box, crop_size=crop_size)
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return crop_rgb, crop_bgr, new_box




