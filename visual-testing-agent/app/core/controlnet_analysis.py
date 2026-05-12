import cv2
import numpy as np
from PIL import Image


def generate_structure_map(image_np):

    resized = cv2.resize(
        image_np,
        (512, 512)
    )

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY
    )

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        blurred,
        80,
        180
    )

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    edges = cv2.dilate(
        edges,
        kernel,
        iterations=1
    )

    edges_rgb = cv2.cvtColor(
        edges,
        cv2.COLOR_GRAY2RGB
    )

    return Image.fromarray(edges_rgb)