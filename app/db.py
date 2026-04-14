import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

#DATABASE_URL = "postgresql://admin:admin123@13.222.10.43:5432/insurance_db"
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://admin:admin123@postgres-db:5432/insurance_db"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


