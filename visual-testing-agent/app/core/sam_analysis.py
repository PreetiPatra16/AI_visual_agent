import numpy as np

from segment_anything import (
    sam_model_registry,
    SamPredictor
)

# ----------------------------------------------------
# LOAD MODEL
# ----------------------------------------------------

sam = sam_model_registry["vit_b"](
    checkpoint="models/sam_vit_b.pth"
)

predictor = SamPredictor(sam)

current_image = None


# ----------------------------------------------------
# INITIALIZE IMAGE
# ----------------------------------------------------

def initialize_sam(image):

    global current_image

    if current_image is not image:

        predictor.set_image(image)

        current_image = image


# ----------------------------------------------------
# REFINE REGION
# ----------------------------------------------------

def refine_region_with_sam(
    x,
    y,
    w,
    h
):

    input_box = np.array([
        x,
        y,
        x + w,
        y + h
    ])

    masks, _, _ = predictor.predict(
        box=input_box,
        multimask_output=False
    )

    mask = masks[0]

    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:

        return x, y, w, h

    nx1 = np.min(xs)
    ny1 = np.min(ys)

    nx2 = np.max(xs)
    ny2 = np.max(ys)

    return (
        nx1,
        ny1,
        nx2 - nx1,
        ny2 - ny1
    )