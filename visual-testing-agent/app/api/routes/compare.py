import os
import json
import base64
import cv2
import numpy as np



from typing import Optional

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException
)

from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool

# ----------------------------------------------------
# CORE
# ----------------------------------------------------

from app.core.image_processor import compare_images

from app.core.summary_generator import (
    generate_final_summary
)

from app.core.detailed_report import (
    generate_detailed_report
)

from app.core.llm_report import (
    generate_llm_report
)

from app.core.controlnet_analysis import (
    generate_structure_map
)

from app.core.structure_comparator import (
    compare_structure
)

# ----------------------------------------------------
# DATABASE
# ----------------------------------------------------

from app.db.database import SessionLocal
from app.db.models import Comparison

# ----------------------------------------------------
# PDF
# ----------------------------------------------------

from services.report import generate_pdf

import os
import json
import base64
import cv2
import numpy as np

import inspect

print("\n===== IMPORT DEBUG =====")

print(
    "SUMMARY FILE:",
    inspect.getfile(generate_final_summary)
)

print(
    "DETAIL REPORT FILE:",
    inspect.getfile(generate_detailed_report)
)

print("========================\n")

from typing import Optional

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException
)

from fastapi.responses import FileResponse
from fastapi.concurrency import run_in_threadpool

# ----------------------------------------------------
# CORE
# ----------------------------------------------------

from app.core.image_processor import compare_images

from app.core.summary_generator import (
    generate_final_summary
)

from app.core.detailed_report import (
    generate_detailed_report
)

from app.core.llm_report import (
    generate_llm_report
)

from app.core.controlnet_analysis import (
    generate_structure_map
)

from app.core.structure_comparator import (
    compare_structure
)

# ----------------------------------------------------
# DATABASE
# ----------------------------------------------------

from app.db.database import SessionLocal
from app.db.models import Comparison

# ----------------------------------------------------
# PDF
# ----------------------------------------------------

from services.report import generate_pdf

# ----------------------------------------------------
# ROUTER
# ----------------------------------------------------

router = APIRouter(
    prefix="/compare",
    tags=["comparison"]
)

# ----------------------------------------------------
# DIRECTORIES
# ----------------------------------------------------

OUTPUT_DIR = "outputs"

UPLOAD_DIR = "storage"

CONTROLNET_OUTPUT_DIR = os.path.join(
    OUTPUT_DIR,
    "controlnet"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CONTROLNET_OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------
# HELPERS
# ----------------------------------------------------

def save_file(file_bytes, filename):

    path = os.path.join(
        UPLOAD_DIR,
        filename
    )

    with open(path, "wb") as f:
        f.write(file_bytes)

    return path


def save_base64_image(base64_str, path):

    with open(path, "wb") as f:
        f.write(base64.b64decode(base64_str))


# ====================================================
# MAIN COMPARISON
# ====================================================

@router.post("/")
async def run_comparison(
    baseline_image: UploadFile = File(...),
    current_image: UploadFile = File(...),
    ignore_region_config: Optional[str] = Form(None)
):

    try:

        allowed_types = {
            "image/png",
            "image/jpeg",
            "image/jpg"
        }

        for file in [baseline_image, current_image]:

            if file.content_type not in allowed_types:

                raise HTTPException(
                    status_code=400,
                    detail="Only PNG/JPG images allowed."
                )

        baseline_bytes = await baseline_image.read()
        current_bytes = await current_image.read()

        ignore_regions = []

        if ignore_region_config:

            ignore_regions = json.loads(
                ignore_region_config
            )

        # ------------------------------------------------
        # IMAGE COMPARISON
        # ------------------------------------------------

        result = await run_in_threadpool(
            compare_images,
            baseline_bytes,
            current_bytes,
            ignore_regions
        )

        comparison_id = result.comparison_id

        # ------------------------------------------------
        # LOAD IMAGES
        # ------------------------------------------------

        baseline_np = cv2.imdecode(
            np.frombuffer(baseline_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        current_np = cv2.imdecode(
            np.frombuffer(current_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        # ------------------------------------------------
        # STRUCTURE MAPS
        # ------------------------------------------------

        baseline_structure = generate_structure_map(
            baseline_np
        )

        current_structure = generate_structure_map(
            current_np
        )

        baseline_structure_path = os.path.join(
            CONTROLNET_OUTPUT_DIR,
            f"{comparison_id}_baseline_structure.png"
        )

        current_structure_path = os.path.join(
            CONTROLNET_OUTPUT_DIR,
            f"{comparison_id}_current_structure.png"
        )

        baseline_structure.save(
            baseline_structure_path
        )

        current_structure.save(
            current_structure_path
        )

        # ------------------------------------------------
        # STRUCTURE SIMILARITY
        # ------------------------------------------------

        baseline_cv = cv2.cvtColor(
            np.array(baseline_structure),
            cv2.COLOR_RGB2BGR
        )

        current_cv = cv2.cvtColor(
            np.array(current_structure),
            cv2.COLOR_RGB2BGR
        )

        structure_score = compare_structure(
            baseline_cv,
            current_cv
        )

        # ------------------------------------------------
        # REPORTS
        # ------------------------------------------------

        summary = generate_final_summary(
            result.regions,
            result.ignored_regions,
            result.image_height,
            result.image_width
        )

        detailed_report = generate_detailed_report(
            result
        )

        llm_report = generate_llm_report(
            result,
            summary,
            detailed_report,
            baseline_bytes,
            current_bytes
        )

        # ------------------------------------------------
        # SAVE OUTPUTS
        # ------------------------------------------------

        baseline_path = save_file(
            baseline_bytes,
            f"{comparison_id}_baseline.png"
        )

        current_path = save_file(
            current_bytes,
            f"{comparison_id}_current.png"
        )

        diff_path = os.path.join(
            OUTPUT_DIR,
            f"{comparison_id}.png"
        )

        save_base64_image(
            result.difference_image_b64,
            diff_path
        )

        # ------------------------------------------------
        # PDF
        # ------------------------------------------------

        pdf_path = os.path.join(
            OUTPUT_DIR,
            f"{comparison_id}.pdf"
        )

        report_text = (

            "SUMMARY:\n"
            + summary

            + "\n\nDETAILED REPORT:\n"
            + "\n".join(detailed_report)

            + "\n\nLLM ANALYSIS:\n"
            + llm_report
        )

        generate_pdf(
            pdf_path,
            diff_path,
            report_text
        )

        # ------------------------------------------------
        # DATABASE
        # ------------------------------------------------

        db = SessionLocal()

        record = Comparison(
            id=comparison_id,
            baseline_path=baseline_path,
            current_path=current_path,
            diff_image_path=diff_path,
            difference_score=result.difference_score,
            summary=summary,
            status="completed"
        )

        db.add(record)

        db.commit()

        db.close()

        # ------------------------------------------------
        # RESPONSE
        # ------------------------------------------------

        return {

            "comparison_id": comparison_id,

            "summary": summary,

            "detailed_report": detailed_report,

            "llm_report": llm_report,

            "ssim_score": result.ssim_score,

            "difference_score": result.difference_score,

            "structure_similarity": structure_score,

            "controlnet_outputs": {

                "baseline_structure":
                    baseline_structure_path,

                "current_structure":
                    current_structure_path
            },

            "download_image":
                f"/compare/image/{comparison_id}",

            "download_report":
                f"/compare/report/{comparison_id}"
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ====================================================
# DOWNLOAD IMAGE
# ====================================================

@router.get("/image/{comparison_id}")
def download_image(comparison_id: str):

    path = os.path.join(
        OUTPUT_DIR,
        f"{comparison_id}.png"
    )

    if not os.path.exists(path):

        raise HTTPException(
            status_code=404,
            detail="Image not found"
        )

    return FileResponse(
        path,
        media_type="image/png",
        filename="result.png"
    )


# ====================================================
# DOWNLOAD REPORT
# ====================================================

@router.get("/report/{comparison_id}")
def download_report(comparison_id: str):

    path = os.path.join(
        OUTPUT_DIR,
        f"{comparison_id}.pdf"
    )

    if not os.path.exists(path):

        raise HTTPException(
            status_code=404,
            detail="Report not found"
        )

    return FileResponse(
        path,
        media_type="application/pdf",
        filename="report.pdf"
    )


# ----------------------------------------------------
# ROUTER
# ----------------------------------------------------

router = APIRouter(
    prefix="/compare",
    tags=["comparison"]
)

# ----------------------------------------------------
# DIRECTORIES
# ----------------------------------------------------

OUTPUT_DIR = "outputs"

UPLOAD_DIR = "storage"

CONTROLNET_OUTPUT_DIR = os.path.join(
    OUTPUT_DIR,
    "controlnet"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CONTROLNET_OUTPUT_DIR, exist_ok=True)

# ----------------------------------------------------
# HELPERS
# ----------------------------------------------------

def save_file(file_bytes, filename):

    path = os.path.join(
        UPLOAD_DIR,
        filename
    )

    with open(path, "wb") as f:
        f.write(file_bytes)

    return path


def save_base64_image(base64_str, path):

    with open(path, "wb") as f:
        f.write(base64.b64decode(base64_str))


# ====================================================
# MAIN COMPARISON
# ====================================================

@router.post("/")
async def run_comparison(
    baseline_image: UploadFile = File(...),
    current_image: UploadFile = File(...),
    ignore_region_config: Optional[str] = Form(None)
):

    try:

        allowed_types = {
            "image/png",
            "image/jpeg",
            "image/jpg"
        }

        for file in [baseline_image, current_image]:

            if file.content_type not in allowed_types:

                raise HTTPException(
                    status_code=400,
                    detail="Only PNG/JPG images allowed."
                )

        baseline_bytes = await baseline_image.read()
        current_bytes = await current_image.read()

        ignore_regions = []

        if ignore_region_config:

            ignore_regions = json.loads(
                ignore_region_config
            )

        # ------------------------------------------------
        # IMAGE COMPARISON
        # ------------------------------------------------

        result = await run_in_threadpool(
            compare_images,
            baseline_bytes,
            current_bytes,
            ignore_regions
        )

        comparison_id = result.comparison_id

        # ------------------------------------------------
        # LOAD IMAGES
        # ------------------------------------------------

        baseline_np = cv2.imdecode(
            np.frombuffer(baseline_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        current_np = cv2.imdecode(
            np.frombuffer(current_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        # ------------------------------------------------
        # STRUCTURE MAPS
        # ------------------------------------------------

        baseline_structure = generate_structure_map(
            baseline_np
        )

        current_structure = generate_structure_map(
            current_np
        )

        baseline_structure_path = os.path.join(
            CONTROLNET_OUTPUT_DIR,
            f"{comparison_id}_baseline_structure.png"
        )

        current_structure_path = os.path.join(
            CONTROLNET_OUTPUT_DIR,
            f"{comparison_id}_current_structure.png"
        )

        baseline_structure.save(
            baseline_structure_path
        )

        current_structure.save(
            current_structure_path
        )

        # ------------------------------------------------
        # STRUCTURE SIMILARITY
        # ------------------------------------------------

        baseline_cv = cv2.cvtColor(
            np.array(baseline_structure),
            cv2.COLOR_RGB2BGR
        )

        current_cv = cv2.cvtColor(
            np.array(current_structure),
            cv2.COLOR_RGB2BGR
        )

        structure_score = compare_structure(
            baseline_cv,
            current_cv
        )

        # ------------------------------------------------
        # REPORTS
        # ------------------------------------------------

        summary = generate_final_summary(
            result.regions,
            result.ignored_regions,
            result.image_height,
            result.image_width
        )

        detailed_report = generate_detailed_report(
            result
        )

        llm_report = generate_llm_report(
            result,
            summary,
            detailed_report,
            baseline_bytes,
            current_bytes
        )

        # ------------------------------------------------
        # SAVE OUTPUTS
        # ------------------------------------------------

        baseline_path = save_file(
            baseline_bytes,
            f"{comparison_id}_baseline.png"
        )

        current_path = save_file(
            current_bytes,
            f"{comparison_id}_current.png"
        )

        diff_path = os.path.join(
            OUTPUT_DIR,
            f"{comparison_id}.png"
        )

        save_base64_image(
            result.difference_image_b64,
            diff_path
        )

        # ------------------------------------------------
        # PDF
        # ------------------------------------------------

        pdf_path = os.path.join(
            OUTPUT_DIR,
            f"{comparison_id}.pdf"
        )

        report_text = (

            "SUMMARY:\n"
            + summary

            + "\n\nDETAILED REPORT:\n"
            + "\n".join(detailed_report)

            + "\n\nLLM ANALYSIS:\n"
            + llm_report
        )

        generate_pdf(
            pdf_path,
            diff_path,
            report_text
        )

        # ------------------------------------------------
        # DATABASE
        # ------------------------------------------------

        db = SessionLocal()

        record = Comparison(
            id=comparison_id,
            baseline_path=baseline_path,
            current_path=current_path,
            diff_image_path=diff_path,
            difference_score=result.difference_score,
            summary=summary,
            status="completed"
        )

        db.add(record)

        db.commit()

        db.close()

        # ------------------------------------------------
        # RESPONSE
        # ------------------------------------------------

        return {

            "comparison_id": comparison_id,

            "summary": summary,

            "detailed_report": detailed_report,

            "llm_report": llm_report,

            "ssim_score": result.ssim_score,

            "difference_score": result.difference_score,

            "structure_similarity": structure_score,

            "controlnet_outputs": {

                "baseline_structure":
                    baseline_structure_path,

                "current_structure":
                    current_structure_path
            },

            "download_image":
                f"/compare/image/{comparison_id}",

            "download_report":
                f"/compare/report/{comparison_id}"
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ====================================================
# DOWNLOAD IMAGE
# ====================================================

@router.get("/image/{comparison_id}")
def download_image(comparison_id: str):

    path = os.path.join(
        OUTPUT_DIR,
        f"{comparison_id}.png"
    )

    if not os.path.exists(path):

        raise HTTPException(
            status_code=404,
            detail="Image not found"
        )

    return FileResponse(
        path,
        media_type="image/png",
        filename="result.png"
    )


# ====================================================
# DOWNLOAD REPORT
# ====================================================

@router.get("/report/{comparison_id}")
def download_report(comparison_id: str):

    path = os.path.join(
        OUTPUT_DIR,
        f"{comparison_id}.pdf"
    )

    if not os.path.exists(path):

        raise HTTPException(
            status_code=404,
            detail="Report not found"
        )

    return FileResponse(
        path,
        media_type="application/pdf",
        filename="report.pdf"
    )