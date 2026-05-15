from collections import Counter


# ====================================================
# POSITION
# ====================================================

def get_position(
    x,
    y,
    img_width,
    img_height
):

    if x < img_width * 0.3:
        h_pos = "left"

    elif x > img_width * 0.7:
        h_pos = "right"

    else:
        h_pos = "center"

    if y < img_height * 0.2:
        v_pos = "top"

    elif y > img_height * 0.8:
        v_pos = "bottom"

    else:
        v_pos = "middle"

    return f"{v_pos}-{h_pos}"


# ====================================================
# GENERATE SUMMARY
# ====================================================

def generate_final_summary(

    regions,

    ignored_count,

    img_height=None,

    img_width=None,

    ssim_score=None,

    difference_score=None,

    structure_similarity=None
):

    summary = []

    # =================================================
    # HEADER
    # =================================================

    summary.append(
        "=== STRUCTURAL VISUAL ANALYSIS REPORT ==="
    )

    summary.append("")

    # =================================================
    # METRICS
    # =================================================

    summary.append(
        "---- Visual Similarity Metrics ----"
    )

    summary.append(
        f"SSIM Similarity Score: "
        f"{ssim_score:.4f}"
    )

    summary.append(
        f"Pixel Difference Score: "
        f"{difference_score:.4f}"
    )

    summary.append(
        f"Structural Layout Similarity: "
        f"{structure_similarity:.4f}"
    )

    summary.append("")

    # =================================================
    # REGION ANALYSIS
    # =================================================

    significant_regions = [

        r for r in regions

        if not r.ignored
    ]

    ignored_regions = [

        r for r in regions

        if r.ignored
    ]

    total_changed_area = 0

    largest_area = 0

    positions = []

    for region in significant_regions:

        box = region.bounding_box

        area = (
            box.width
            * box.height
        )

        total_changed_area += area

        largest_area = max(
            largest_area,
            area
        )

        positions.append(

            get_position(
                box.x,
                box.y,
                img_width,
                img_height
            )
        )

    avg_area = 0

    if significant_regions:

        avg_area = (
            total_changed_area
            / len(significant_regions)
        )

    # =================================================
    # STRUCTURAL METRICS
    # =================================================

    summary.append(
        "---- Structural Change Metrics ----"
    )

    summary.append(
        f"Total Significant Regions: "
        f"{len(significant_regions)}"
    )

    summary.append(
        f"Ignored Dynamic Regions: "
        f"{len(ignored_regions)}"
    )

    summary.append(
        f"Total Changed Area: "
        f"{total_changed_area}px"
    )

    summary.append(
        f"Largest Region Area: "
        f"{largest_area}px"
    )

    summary.append(
        f"Average Region Area: "
        f"{int(avg_area)}px"
    )

    summary.append("")

    # =================================================
    # REGION DISTRIBUTION
    # =================================================

    summary.append(
        "---- UI Distribution Analysis ----"
    )

    if positions:

        counts = Counter(positions)

        for pos, count in counts.items():

            summary.append(
                f"{count} modified region(s) "
                f"detected in the "
                f"{pos} section"
            )

    else:

        summary.append(
            "No significant UI distribution changes detected."
        )

    summary.append("")

    # =================================================
    # COMPLEXITY ANALYSIS
    # =================================================

    summary.append(
        "---- Structural Complexity Analysis ----"
    )

    if structure_similarity > 0.90:

        complexity = (
            "Minimal structural deviation"
        )

    elif structure_similarity > 0.75:

        complexity = (
            "Moderate structural modification"
        )

    else:

        complexity = (
            "High structural divergence"
        )

    summary.append(
        f"Layout Complexity Assessment: "
        f"{complexity}"
    )

    # ------------------------------------------------

    if total_changed_area > 200000:

        density = "High"

    elif total_changed_area > 80000:

        density = "Moderate"

    else:

        density = "Low"

    summary.append(
        f"Visual Change Density: "
        f"{density}"
    )

    # ------------------------------------------------

    if significant_regions:

        summary.append(
            "Detected UI modifications "
            "are concentrated in localized "
            "interface sections rather than "
            "spread across the entire layout."
        )

    else:

        summary.append(
            "No meaningful structural "
            "layout changes detected."
        )

    summary.append("")

    # =================================================
    # CONTROLNET ANALYSIS
    # =================================================

    summary.append(
        "---- ControlNet Structural Interpretation ----"
    )

    if structure_similarity < 0.70:

        summary.append(
            "Significant structural "
            "variation detected between "
            "baseline and current layouts."
        )

    elif structure_similarity < 0.90:

        summary.append(
            "Moderate layout-level "
            "structural modifications detected."
        )

    else:

        summary.append(
            "Layout structure remains "
            "highly consistent between "
            "both UI versions."
        )

    if significant_regions:

        summary.append(
            "Structural edge-map analysis "
            "indicates new interface "
            "components or layout expansion "
            "within the modified regions."
        )

    summary.append("")

    # =================================================
    # IGNORED REGIONS
    # =================================================

    if ignored_regions:

        summary.append(
            "---- Ignored Dynamic Regions ----"
        )

        for i, region in enumerate(

            ignored_regions,

            start=1
        ):

            box = region.bounding_box

            summary.append(
                f"{i}. Dynamic region ignored "
                f"at x={box.x}, "
                f"y={box.y}, "
                f"size={box.width}x{box.height}"
            )

        summary.append("")

    # =================================================
    # FINAL SYSTEM INSIGHT
    # =================================================

    summary.append(
        "---- System-Level Structural Insight ----"
    )

    if significant_regions:

        summary.append(
            "The detected visual modifications "
            "primarily affect localized UI "
            "workflow sections while preserving "
            "overall application layout stability."
        )

    else:

        summary.append(
            "The compared interfaces remain "
            "structurally consistent with "
            "minimal semantic deviation."
        )

    return "\n".join(summary)