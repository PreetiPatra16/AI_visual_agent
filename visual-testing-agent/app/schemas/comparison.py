from pydantic import BaseModel
from typing import List
from datetime import datetime
from typing import Optional

class BoundingBox(BaseModel):
    x: int
    y: int
    width: int
    height: int


class DifferenceRegion(BaseModel):
    bounding_box: BoundingBox
    area: float
    label: str
    ignored: bool = False   # for dynamic ignored regions


class ComparisonResult(BaseModel):
    comparison_id: str
    difference_score: float
    ssim_score: float
    total_regions: int
    ignored_regions: int    # count of ignored regions
    regions: List[DifferenceRegion]
    difference_image_b64: str
    image_width: int
    image_height: int
    status: str
    timestamp: datetime
    summary: Optional[str] = None
    psnr: float
    snr: float
    diff_mean: float
    diff_std: float