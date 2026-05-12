import cv2
import numpy as np
import base64
import uuid
import pytesseract

from datetime import datetime, timezone
from typing import Optional

from skimage.metrics import (
    structural_similarity as ssim
)

from app.schemas.comparison import (
    ComparisonResult,
    DifferenceRegion,
    BoundingBox
)

from app.core.sam_analysis import (
    initialize_sam,
    refine_region_with_sam
)

# ====================================================
# CONFIG
# ====================================================

MIN_CONTOUR_AREA = 200

MIN_WIDTH = 40
MIN_HEIGHT = 40

MAX_ASPECT_RATIO = 10
MIN_ASPECT_RATIO = 0.1

MERGE_DISTANCE = 60

# ====================================================
# IMAGE LOADING
# ====================================================

def load_image_from_bytes(
    data: bytes
):

    arr = np.frombuffer(
        data,
        np.uint8
    )

    image = cv2.imdecode(
        arr,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise ValueError(
            "Invalid image."
        )

    return image


# ====================================================
# VALIDATION
# ====================================================

def validate_image(
    image,
    name="image"
):

    h, w = image.shape[:2]

    if w < 50 or h < 50:

        raise ValueError(
            f"{name} too small."
        )

    if w > 7680 or h > 4320:

        raise ValueError(
            f"{name} too large."
        )


# ====================================================
# RESIZE
# ====================================================

def resize_to_common(
    img1,
    img2
):

    h, w = img1.shape[:2]

    img2 = cv2.resize(
        img2,
        (w, h)
    )

    return img1, img2


# ====================================================
# IMAGE ALIGNMENT
# ====================================================

def align_images(
    img1,
    img2
):

    gray1 = cv2.cvtColor(
        img1,
        cv2.COLOR_BGR2GRAY
    )

    gray2 = cv2.cvtColor(
        img2,
        cv2.COLOR_BGR2GRAY
    )

    orb = cv2.ORB_create(1000)

    kp1, des1 = orb.detectAndCompute(
        gray1,
        None
    )

    kp2, des2 = orb.detectAndCompute(
        gray2,
        None
    )

    if des1 is None or des2 is None:
        return img2

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True
    )

    matches = matcher.match(
        des1,
        des2
    )

    if len(matches) < 8:
        return img2

    matches = sorted(
        matches,
        key=lambda x: x.distance
    )[:100]

    src_pts = np.float32([
        kp2[m.trainIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    dst_pts = np.float32([
        kp1[m.queryIdx].pt
        for m in matches
    ]).reshape(-1, 1, 2)

    matrix, _ = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        5.0
    )

    if matrix is None:
        return img2

    h, w = img1.shape[:2]

    aligned = cv2.warpPerspective(
        img2,
        matrix,
        (w, h)
    )

    return aligned


# ====================================================
# DIFFERENCE DETECTION
# ====================================================

def detect_difference(
    img1,
    img2
):

    gray1 = cv2.cvtColor(
        img1,
        cv2.COLOR_BGR2GRAY
    )

    gray2 = cv2.cvtColor(
        img2,
        cv2.COLOR_BGR2GRAY
    )

    score, diff = ssim(
        gray1,
        gray2,
        full=True
    )

    diff = (
        np.clip(diff, 0, 1) * 255
    ).astype("uint8")

    diff = cv2.GaussianBlur(
        diff,
        (5, 5),
        0
    )

    thresh = cv2.threshold(
        diff,
        0,
        255,
        cv2.THRESH_BINARY_INV |
        cv2.THRESH_OTSU
    )[1]

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    thresh = cv2.dilate(
        thresh,
        kernel,
        iterations=2
    )

    return thresh, score


# ====================================================
# MERGE NEARBY BOXES
# ====================================================

def merge_nearby_boxes(
    boxes,
    distance=MERGE_DISTANCE
):

    merged = []

    while boxes:

        x, y, w, h = boxes.pop(0)

        current = [
            x,
            y,
            x + w,
            y + h
        ]

        changed = True

        while changed:

            changed = False

            remaining = []

            for bx, by, bw, bh in boxes:

                if (

                    bx < current[2] + distance
                    and

                    bx + bw > current[0] - distance
                    and

                    by < current[3] + distance
                    and

                    by + bh > current[1] - distance
                ):

                    current[0] = min(
                        current[0],
                        bx
                    )

                    current[1] = min(
                        current[1],
                        by
                    )

                    current[2] = max(
                        current[2],
                        bx + bw
                    )

                    current[3] = max(
                        current[3],
                        by + bh
                    )

                    changed = True

                else:

                    remaining.append(
                        (bx, by, bw, bh)
                    )

            boxes = remaining

        merged.append((
            current[0],
            current[1],
            current[2] - current[0],
            current[3] - current[1]
        ))

    return merged


# ====================================================
# DYNAMIC REGION DETECTION
# ====================================================

def is_dynamic_region(
    region,
    y,
    total_height
):

    try:

        gray = cv2.cvtColor(
            region,
            cv2.COLOR_BGR2GRAY
        )

        # STATUS BAR

        if y < total_height * 0.06:
            return True

        # VERY FLAT

        if np.std(gray) < 5:
            return True

        text = pytesseract.image_to_string(
            gray
        ).lower()

        keywords = [
            "battery",
            "network",
            "time",
            "profile",
            "welcome"
        ]

        for word in keywords:

            if word in text:
                return True

    except:
        pass

    return False


# ====================================================
# OCR HELPER
# ====================================================

def extract_text(image):

    image = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY +
        cv2.THRESH_OTSU
    )[1]

    text = pytesseract.image_to_string(
        image
    )

    return text.strip()


# ====================================================
# CLASSIFY REGION
# ====================================================

def classify_region(
    baseline,
    current,
    x,
    y,
    w,
    h
):

    r1 = baseline[y:y+h, x:x+w]
    r2 = current[y:y+h, x:x+w]

    if r1.size == 0 or r2.size == 0:
        return "Invalid Region"

    try:

        g1 = cv2.cvtColor(
            r1,
            cv2.COLOR_BGR2GRAY
        )

        g2 = cv2.cvtColor(
            r2,
            cv2.COLOR_BGR2GRAY
        )

    except:

        return "Processing Error"

    text1 = extract_text(g1)
    text2 = extract_text(g2)

    try:

        score, _ = ssim(
            g1,
            g2,
            full=True
        )

    except:

        score = 0.0

    mean1 = np.mean(r1)
    mean2 = np.mean(r2)

    # TEXT CHANGE

    if text1 != text2 and (text1 or text2):

        if len(text2) > len(text1):

            return "New Text / Section Added"

        return "Text Updated"

    # THEME CHANGE

    if abs(mean1 - mean2) > 40:

        return "Theme or Color Updated"

    # LAYOUT CHANGE

    if score < 0.40:

        return "Layout Modified"

    # COMPONENT CHANGE

    if score < 0.75:

        return "Component Updated"

    return "Minor Visual Adjustment"


# ====================================================
# METRICS
# ====================================================

def compute_psnr(
    img1,
    img2
):

    mse = np.mean(
        (img1 - img2) ** 2
    )

    if mse == 0:
        return 100

    return 20 * np.log10(
        255.0 / np.sqrt(mse)
    )


def compute_snr(
    img1,
    img2
):

    signal = np.mean(
        img1 ** 2
    )

    noise = np.mean(
        (img1 - img2) ** 2
    )

    if noise == 0:
        return float("inf")

    return 10 * np.log10(
        signal / noise
    )


# ====================================================
# MAIN PIPELINE
# ====================================================

def compare_images(
    baseline_bytes: bytes,
    current_bytes: bytes,
    ignore_regions: Optional[list] = None
):

    # ------------------------------------------------
    # LOAD
    # ------------------------------------------------

    baseline = load_image_from_bytes(
        baseline_bytes
    )

    current = load_image_from_bytes(
        current_bytes
    )

    validate_image(
        baseline,
        "baseline"
    )

    validate_image(
        current,
        "current"
    )

    # ------------------------------------------------
    # PREPROCESS
    # ------------------------------------------------

    baseline, current = resize_to_common(
        baseline,
        current
    )

    current = align_images(
        baseline,
        current
    )

    initialize_sam(current)

    # ------------------------------------------------
    # DIFFERENCE
    # ------------------------------------------------

    thresh, score = detect_difference(
        baseline,
        current
    )

    # ------------------------------------------------
    # METRICS
    # ------------------------------------------------

    psnr = compute_psnr(
        baseline,
        current
    )

    snr = compute_snr(
        baseline,
        current
    )

    diff = cv2.absdiff(
        baseline,
        current
    )

    diff_gray = cv2.cvtColor(
        diff,
        cv2.COLOR_BGR2GRAY
    )

    diff_mean = float(
        np.mean(diff_gray)
    )

    diff_std = float(
        np.std(diff_gray)
    )

    # ------------------------------------------------
    # IGNORE REGIONS
    # ------------------------------------------------

    ignore_regions_tuples = []

    if ignore_regions:

        for r in ignore_regions:

            x = r["x"]
            y = r["y"]
            w = r["width"]
            h = r["height"]

            thresh[y:y+h, x:x+w] = 0

            ignore_regions_tuples.append(
                (x, y, w, h)
            )

    # ------------------------------------------------
    # FIND CONTOURS
    # ------------------------------------------------

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < MIN_CONTOUR_AREA:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        if w < MIN_WIDTH or h < MIN_HEIGHT:
            continue

        aspect_ratio = w / float(h)

        if (
            aspect_ratio > MAX_ASPECT_RATIO
            or
            aspect_ratio < MIN_ASPECT_RATIO
        ):
            continue

        boxes.append(
            (x, y, w, h)
        )

    # ------------------------------------------------
    # MERGE BOXES
    # ------------------------------------------------

    boxes = merge_nearby_boxes(
        boxes
    )

    # ------------------------------------------------
    # OUTPUTS
    # ------------------------------------------------

    regions = []

    ignored_count = 0

    diff_image = cv2.addWeighted(
        baseline,
        0.5,
        current,
        0.5,
        0
    )

    # ------------------------------------------------
    # PROCESS REGIONS
    # ------------------------------------------------

    for (x, y, w, h) in boxes:

        x, y, w, h = refine_region_with_sam(
            x,
            y,
            w,
            h
        )

        region = current[
            y:y+h,
            x:x+w
        ]

        ignored = False

        # --------------------------------------------
        # MANUAL IGNORE
        # --------------------------------------------

        for (
            ix,
            iy,
            iw,
            ih
        ) in ignore_regions_tuples:

            if (
                x >= ix
                and
                y >= iy
                and
                (x+w) <= (ix+iw)
                and
                (y+h) <= (iy+ih)
            ):

                ignored = True
                break

        # --------------------------------------------
        # DYNAMIC REGION
        # --------------------------------------------

        if ignored:

            label = "Ignored Dynamic Region"

            ignored_count += 1

            color = (128, 128, 128)

        elif is_dynamic_region(
            region,
            y,
            baseline.shape[0]
        ):

            label = "Ignored Dynamic Region"

            ignored = True

            ignored_count += 1

            color = (128, 128, 128)

        else:

            label = classify_region(
                baseline,
                current,
                x,
                y,
                w,
                h
            )

            color = (0, 255, 0)

        # --------------------------------------------
        # DRAW
        # --------------------------------------------

        cv2.rectangle(
            diff_image,
            (x, y),
            (x+w, y+h),
            color,
            2
        )

        # --------------------------------------------
        # STORE
        # --------------------------------------------

        regions.append(

            DifferenceRegion(

                bounding_box=BoundingBox(
                    x=x,
                    y=y,
                    width=w,
                    height=h
                ),

                area=float(w * h),

                label=label,

                ignored=ignored
            )
        )

    # ------------------------------------------------
    # ENCODE IMAGE
    # ------------------------------------------------

    diff_b64 = base64.b64encode(
        cv2.imencode(
            ".png",
            diff_image
        )[1]
    ).decode()

    # ------------------------------------------------
    # FINAL RESPONSE
    # ------------------------------------------------

    return ComparisonResult(

        comparison_id=str(uuid.uuid4()),

        difference_score=round(
            1 - score,
            4
        ),

        ssim_score=round(
            score,
            4
        ),

        total_regions=len(regions),

        ignored_regions=ignored_count,

        regions=regions,

        difference_image_b64=diff_b64,

        image_width=baseline.shape[1],

        image_height=baseline.shape[0],

        status="completed",

        timestamp=datetime.now(
            timezone.utc
        ),

        psnr=round(psnr, 4),

        snr=round(snr, 4),

        diff_mean=round(diff_mean, 4),

        diff_std=round(diff_std, 4)
    )