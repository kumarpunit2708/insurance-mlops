from app.db import engine
from app import models  # 👈 IMPORTANT (this loads models)

models.Base.metadata.create_all(bind=engine)

print("Tables created successfully ✅")