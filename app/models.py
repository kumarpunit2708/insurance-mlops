from sqlalchemy import Column, Integer, Float, String, TIMESTAMP
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    age = Column(Integer)
    bmi = Column(Float)
    children = Column(Integer)
    sex = Column(String)
    smoker = Column(String)
    region = Column(String)
    prediction = Column(Float)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)