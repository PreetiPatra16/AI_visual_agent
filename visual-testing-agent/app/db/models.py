from sqlalchemy import Column, String, Float, DateTime, Text
from datetime import datetime
from app.db.database import Base


class Comparison(Base):
    __tablename__ = "comparisons"

    id = Column(String, primary_key=True)

    baseline_path = Column(String)
    current_path = Column(String)
    diff_image_path = Column(String)

    difference_score = Column(Float)
    summary = Column(Text)

    timestamp = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="completed")