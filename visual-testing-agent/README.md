# Visual Testing Agent

An AI-powered Visual Testing Agent designed to detect, classify, and manage UI visual regressions.

## Project Status

Currently, the project has successfully completed **Phase 1: Basic Visual Comparison Engine** and **Phase 2: AI-Based Difference Classification**.

### Phase 1 Features:
* **Image Input Handling:** Accepts Baseline & Current image types seamlessly (PNG/JPEG). Validates structural sizing boundaries.
* **Intelligent Preprocessing:** Aligns image sets proactively via ORB Feature Maps + Homography to handle small scroll displacements and automatically enforces resolution parity before running pixel comparison mapping.
* **Precise Difference Detection:** Uses OpenCV + SSIM (Structural Similarity Index) mapped inside grayscale to detect bounding box regions of discrepancies, ignoring standard minor pixel noises and user-defined localized "ignored regions". Returns actionable metrics securely and highlights visual changes via color-coded overlapping bounds bounds.

### Phase 2 Features:
* **AI-Based Classification:** Each detected difference region is automatically classified into categories like "Text Changed", "Button Missing", "Image Changed", "Misalignment", "Font/Color Change", "New Unexpected Element", or "Dynamic Data Ignored".
* **Smart Dynamic Region Exclusion:** Automatically detects and ignores regions containing dynamic content (dates, times, user data, login info) using OCR-based keyword detection.
* **Color-Coded Visual Feedback:** Difference bounding boxes are color-coded by change type for immediate visual identification.

## Implementation Steps

### Step 1: Image Input Handling
**Files involved:**
- `app/api/routes/compare.py`: Handles HTTP API endpoint (`POST /compare`) that accepts baseline and current images as `UploadFile` objects. Validates MIME types to ensure PNG/JPEG support only.
- `app/core/image_processor.py`: 
  - `load_image_from_bytes()` function decodes uploaded bytes into OpenCV BGR ndarrays using `cv2.imdecode()`
  - `validate_image()` function checks image dimensions (min 50x50, max 7680x4320) and ensures successful loading

### Step 2: Image Preprocessing  
**File: `app/core/image_processor.py`**
- `resize_to_common()`: Resizes both images to match the baseline image's resolution using `cv2.resize()` with INTER_AREA interpolation
- `align_images()`: Uses ORB feature detection + homography transformation (`cv2.findHomography()`) to handle minor screen shifts (scrolling, positioning differences)
- Grayscale conversion: `cv2.cvtColor()` converts both images to grayscale before SSIM comparison

### Step 3: Pixel Difference Detection
**Files involved:**
- `app/core/image_processor.py`: 
  - Uses `skimage.metrics.structural_similarity()` (SSIM) to generate a difference heatmap
  - Applies Otsu thresholding (`cv2.threshold()`) to create a binary mask
  - Contour detection with `cv2.findContours()` to identify changed regions
  - Draws red bounding boxes (`cv2.rectangle()`) and green contour overlays on the difference image
- `app/schemas/comparison.py`: Defines data structures (`BoundingBox`, `DifferenceRegion`, `ComparisonResult`) to store detected differences with coordinates and areas

### Step 4: Difference Categorization (Phase 2)
**File: `app/core/image_processor.py`**
- `classify_region()`: Uses OCR (Tesseract) and image analysis to classify each difference region into:
  - "Text Changed" (OCR detects different text)
  - "Font/Color Change" (brightness difference > 25)
  - "Image Changed" (SSIM score < 0.4)
  - "Misalignment" (region size mismatch)
  - "Button Missing" (region becomes very dark)
  - "New Unexpected Element" (baseline region is very dark)
  - "Unknown" (fallback category)

### Step 5: Dynamic Region Exclusion (Phase 2)
**File: `app/core/image_processor.py`**
- `is_dynamic_region()`: Uses OCR to scan regions for dynamic keywords ("date", "time", "user", "login", "today")
- Automatically marks matching regions as "Dynamic Data Ignored" instead of classifying them
- Prevents false positives from changing timestamps, user data, etc.

The core processing pipeline is orchestrated in the `compare_images()` function in `image_processor.py`, which is called from the API route in `compare.py`. The FastAPI app setup is in `main.py`.

---

## Codebase Map: What is done where?

The repository utilizes modern Python practices via **FastAPI** as the backend framework, **Pydantic** for standard data-classing schemas, and **OpenCV / skimage** for rigorous visual calculations.

### `main.py`
The FastAPI application entry point. 
- Defines the `FastAPI` instance configuration.
- Bootstraps CORS routing middleware.
- Connects the central router prefix (`/api/v1`) to the image comparison APIs.
- Carries a generic `/health` operational ping check.

### `app/api/routes/compare.py`
Handles all HTTP exposure relating to operations.
- `POST /compare` (UploadFile endpoint): Consumes image blobs arrays and dynamically converts frontend variables directly towards threading pools for processing validation (`app.core.image_processor`). 
- `GET /compare/{comparison_id}/image`: Helper cache route that streams a downloadable `.png` of the generated difference mapping (highlighted in red contours).

### `app/core/image_processor.py`
The heavy-lifting calculation logic block.
- **Loading & Validation:** Uses `load_image_from_bytes` & `validate_image` to safely parse bytes streams to `cv2` readable forms preventing corrupted input.
- **Preprocessing:** `resize_to_common()` scales targets perfectly against the baseline image. `align_images()` utilizes RANSAC + `cv2.ORB_create()` mappings to correct minor layout-scroll shifting faults.
- **Engine Logic (`compare_images`):** Orchestrates the total calculation step. Performs grayscale reduction, passes through `skimage`'s `ssim` functionality for scoring mapping thresholds automatically using `cv2.THRESH_OTSU` logic to formulate contours against pixel differences. Finally drops in the user-ignored boxes directly mapping over `cv2.findContours` external traces. Builds an overlaid red + green visual image encoded directly back as base64 string. 

### `app/schemas/comparison.py`
The Pydantic data structuring logic models.
- **`ComparisonResult`:** Represents the structural definition returned entirely from calculation mapping ensuring typing safety (contains IDs, SSIM scores, height/width integers, timestamps and the base64 mapping).
- **`DifferenceRegion` / `BoundingBox`:** Contains structured geometric data plotting coordinates to represent precisely which parts of the interface experienced differences natively.

--- 

AI-Based Visual Testing Agent

Objective

Build an AI agent that can:

* Compare baseline and latest screenshots

* Detect UI differences

* Classify differences as:

  * Layout change

  * Text change

  * Missing element

  * Color/style change

  * Alignment issue

  * Dynamic content change

* Generate human-readable defect summary

* Expose results through API for future integration with any Automation Tool

---

Phase 1: Basic Visual Comparison Engine

**Step 1: Image Input Handling**

* Accept:

  * Baseline image

  * Current image

* Support PNG/JPEG

* Validate image size and format

**Step 2: Image Preprocessing**

* Resize images to common resolution

* Convert to grayscale if needed

* Align images before comparison

* Handle minor screen shifts

**Step 3: Pixel Difference Detection**

* Use:

  * OpenCV

  * SSIM (Structural Similarity Index)

  * Contour detection

* Highlight changed regions with bounding boxes

---

Phase 2: AI-Based Difference Classification

**Step 4: Difference Categorization** For each detected region, classify issue into:

* Text changed

* Button missing

* Image changed

* Misalignment

* Font/color change

* New unexpected element

* Dynamic data ignored

**Step 5: Dynamic Region Exclusion** Allow marking some regions as:

* Ignore dynamic date/time

* Ignore ads/banners

* Ignore rotating carousels

* Ignore user-specific data

---

Phase 3: AI Summary Generation

**Step 6: Generate Natural Language Summary** Agent should produce output like:

Visual comparison completed. 3 significant UI differences found: 1\. Login button missing from header section 2\. Profile icon shifted slightly to the left 3\. Footer background color changed from blue to gray 2 dynamic changes ignored: 1\. Current date field. 2\. Notification count badge

**Step 7: Create FastAPI endpoints**

1\. /compare (POST)

* **Input:**

  * baseline image: Image file (PNG/JPEG)

  * current image: Image file (PNG/JPEG)

  * ignore\_region\_config: JSON (optional)

* **Output:**

  * comparison\_id: string

  * difference\_score: float

  * difference\_image: base64 string

  * difference summary: string

  * difference\_category\_list: array

2\. /results/{id} (GET)

* **Path Parameter:**

  * id: Comparison ID

* **Output:**

  * comparison id: string

  * timestamp: datetime

  * difference score: float

  * difference image: base64 string

  * difference summary: string

  * difference\_category\_list: array

  * baseline\_image path: string

  * current\_image\_path: string

  * status: string

3\. /config/ignore-regions (POST/GET)

* **Input (POST):**

  * ignore\_regions: array of objects

  * x: integer

  * y: integer

  * width: integer

  * height: integer

  * label: string

  * reason: string (optional)

* **Output (POST):**

  * status: string

  * updated regions: array

  * region\_count: integer

* **Output (GET):**

  * ignore\_regions: array

  * region count: integer

  * last updated: datetime

---

Phase 5: Storage and Reporting

**Step 8: Store Results** Save:

* Baseline image path

* Current image path

* Difference image

* Timestamp

* Summary

* Difference percentage

* Use SQLite or PostgreSQL.

---

Deliverables

* Architecture diagram

* Working source code

* API collection (Postman)

* README

* Demo video  
