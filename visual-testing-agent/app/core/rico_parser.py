import json


# ====================================================
# LOAD RICO JSON
# ====================================================

def load_rico_annotation(
    file_path
):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        return json.load(f)


# ====================================================
# EXTRACT COMPONENTS
# ====================================================

def extract_components(
    node,
    components=None
):

    if components is None:
        components = []

    # ------------------------------------------------
    # COMPONENT TYPE
    # ------------------------------------------------

    component_type = node.get(
        "componentLabel"
    )

    bounds = node.get(
        "bounds"
    )

    # ------------------------------------------------
    # STORE ONLY VALID COMPONENTS
    # ------------------------------------------------

    if component_type and bounds:

        components.append({

            "type":
                component_type,

            "bounds":
                bounds
        })

    # ------------------------------------------------
    # CHILDREN
    # ------------------------------------------------

    children = node.get(
        "children",
        []
    )

    for child in children:

        extract_components(
            child,
            components
        )

    return components