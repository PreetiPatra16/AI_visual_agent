from app.core.rico_parser import (
    load_rico_annotation,
    extract_components
)

# ====================================================
# LOAD COMPLEX SAMPLE
# ====================================================

annotation = load_rico_annotation(
    "datasets/rico/semantic_annotations/10.json"
)

# ====================================================
# EXTRACT COMPONENTS
# ====================================================

components = extract_components(
    annotation
)

# ====================================================
# PRINT RESULTS
# ====================================================

print(
    f"\nLoaded {len(components)} components\n"
)

for component in components[:20]:

    print(component)