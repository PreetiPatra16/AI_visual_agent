import cv2
import numpy as np

from skimage.metrics import (
    structural_similarity as ssim
)


def compare_structure(img1, img2):

    g1 = cv2.cvtColor(
        img1,
        cv2.COLOR_BGR2GRAY
    )

    g2 = cv2.cvtColor(
        img2,
        cv2.COLOR_BGR2GRAY
    )

    ssim_score, _ = ssim(
        g1,
        g2,
        full=True
    )

    intersection = np.logical_and(
        g1 > 0,
        g2 > 0
    )

    union = np.logical_or(
        g1 > 0,
        g2 > 0
    )

    iou_score = (
        np.sum(intersection)
        /
        np.sum(union)
    )

    final_score = (
        0.6 * ssim_score
        +
        0.4 * iou_score
    )

    return round(final_score, 4)