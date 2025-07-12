import os
from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Используем переменную окружения DATABASE_URL или дефолтное значение
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://user:password@localhost:5432/comclassify2"
)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    comment = Column(String, nullable=False)
    predict_class = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


# Создать таблицы (вызывать один раз при старте)
def init_db():
    Base.metadata.create_all(bind=engine)
