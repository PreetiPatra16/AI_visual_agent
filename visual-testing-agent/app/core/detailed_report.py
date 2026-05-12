from collections import Counter


# ----------------------------------------------------
# OBJECT SIZE
# ----------------------------------------------------

def classify_object(width, height):

    area = width * height

    if area < 2000:
        return "small component"

    elif area < 10000:
        return "medium component"

    return "large section"


# ----------------------------------------------------
# POSITION
# ----------------------------------------------------

def describe_position(x, y, img_w, img_h):

    if x < img_w * 0.3:
        h_pos = "left"

    elif x > img_w * 0.7:
        h_pos = "right"

    else:
        h_pos = "center"

    if y < img_h * 0.2:
        v_pos = "top"

    elif y > img_h * 0.8:
        v_pos = "bottom"

    else:
        v_pos = "middle"

    return f"{v_pos}-{h_pos}"


# ----------------------------------------------------
# REGION DETAIL
# ----------------------------------------------------

def generate_region_detail(
    region,
    img_w,
    img_h
):

    bbox = region.bounding_box

    x = bbox.x
    y = bbox.y
    w = bbox.width
    h = bbox.height

    position = describe_position(
        x,
        y,
        img_w,
        img_h
    )

    return (
        f"{region.label} detected "
        f"in the {position} section"
    )


# ----------------------------------------------------
# FINAL REPORT
# ----------------------------------------------------

def generate_detailed_report(result):

    regions = result.regions

    report = []

    report.append(
        "=== AI VISUAL TESTING REPORT ==="
    )

    report.append(
        f"SSIM Score: {result.ssim_score:.4f}"
    )

    report.append(
        f"Difference Score: "
        f"{result.difference_score:.4f}"
    )

    report.append(
        f"Regions Detected: "
        f"{result.total_regions}"
    )

    report.append(
        f"Ignored Regions: "
        f"{result.ignored_regions}"
    )

    report.append("")

    # ------------------------------------------------
    # SUMMARY
    # ------------------------------------------------

    report.append(
        "---- Change Summary ----"
    )

    labels = [
        r.label
        for r in regions
    ]

    counts = Counter(labels)

    for label, count in counts.items():

        report.append(
            f"{label}: {count}"
        )

    report.append("")

    # ------------------------------------------------
    # SIGNIFICANT CHANGES
    # ------------------------------------------------

    report.append(
        "---- Significant Changes ----"
    )

    seen = set()

    count = 1

    for region in regions:

        if region.ignored:
            continue

        description = generate_region_detail(
            region,
            result.image_width,
            result.image_height
        )

        if description not in seen:

            seen.add(description)

            report.append(
                f"{count}. {description}"
            )

            count += 1

    if count == 1:

        report.append(
            "No major UI changes detected."
        )

    return report