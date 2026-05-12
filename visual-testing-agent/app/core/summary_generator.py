from collections import Counter

# ----------------------------------------------------
# POSITION DESCRIPTION
# ----------------------------------------------------

def get_position_desc(region, img_height, img_width):

    x = region.bounding_box.x
    y = region.bounding_box.y

    if y < img_height * 0.2:
        section = "header section"

    elif y > img_height * 0.8:
        section = "footer section"

    else:
        section = "main content area"

    return section


# ----------------------------------------------------
# SEMANTIC REGION DESCRIPTION
# ----------------------------------------------------

def generate_region_sentence(
    region,
    img_height,
    img_width
):

    label = region.label.lower()

    section = get_position_desc(
        region,
        img_height,
        img_width
    )

    # TEXT CHANGES

    if "text changed" in label:

        return f"Text content updated in {section}"

    # CONTENT CHANGES

    elif "content changed" in label:

        return f"UI content modified in {section}"

    # STRUCTURE CHANGES

    elif "structure changed" in label:

        return f"Layout structure modified in {section}"

    # COLOR CHANGES

    elif "color change" in label:

        return f"Visual styling changed in {section}"

    # MINOR CHANGES

    elif "minor change" in label:

        return f"Minor UI difference detected in {section}"

    return None


# ----------------------------------------------------
# HEURISTIC SUMMARY
# ----------------------------------------------------

def generate_heuristic_summary(
    regions,
    ignored_count,
    img_height,
    img_width
):

    significant_regions = [
        r for r in regions
        if not r.ignored
    ]

    summary_lines = []

    summary_lines.append(
        "Visual comparison completed."
    )

    meaningful = []

    for region in significant_regions:

        sentence = generate_region_sentence(
            region,
            img_height,
            img_width
        )

        if sentence:

            meaningful.append(sentence)

    # REMOVE DUPLICATES

    meaningful = list(
        dict.fromkeys(meaningful)
    )

    if meaningful:

        summary_lines.append(
            f"{len(meaningful)} meaningful UI changes found:"
        )

        for i, sentence in enumerate(
            meaningful[:5],
            start=1
        ):

            summary_lines.append(
                f"{i}. {sentence}"
            )

    else:

        summary_lines.append(
            "No meaningful UI differences found."
        )

    if ignored_count > 0:

        summary_lines.append(
            f"{ignored_count} dynamic regions ignored."
        )

    return "\n".join(summary_lines)


# ----------------------------------------------------
# FINAL SUMMARY
# ----------------------------------------------------

def generate_final_summary(
    regions,
    ignored_count,
    img_height=None,
    img_width=None
):

    if img_height is None:
        img_height = 1000

    if img_width is None:
        img_width = 1000

    return generate_heuristic_summary(
        regions,
        ignored_count,
        img_height,
        img_width
    )