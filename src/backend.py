import logging
import pickle

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError

from database import Prediction, SessionLocal, init_db


logging.basicConfig(level=logging.INFO)
logging.info("Инициализирую приложение FastApi")

LR_MODEL_PATH = "models/logreg_adv.pkl"
TFIDF_PATH = "models/tfidf_adv.pkl"
ID2TEXT = {0: "негативная", 1: "позитивная", 2: "нейтральная", 3: "мусор"}

app = FastAPI()


@app.on_event("startup")
async def on_startup():
    init_db()


def load_models():
    logging.info("Loading models")
    with open(LR_MODEL_PATH, "rb") as f:
        lr_model = pickle.load(f)
    with open(TFIDF_PATH, "rb") as f:
        tfidf = pickle.load(f)
    logging.info("Models loaded success")
    return lr_model, tfidf

lr_model, tfidf = load_models()

class TextRequest(BaseModel):
    text: str


@app.post("/predict/")
async def predict(request: TextRequest):
    logging.info(f"Recieved text for prediction: {request.text}")
    text = request.text
    features = tfidf.transform([text])
    logging.info("Making prediction")
    predict_class = lr_model.predict(features)[0]
    predict_class_text = ID2TEXT[int(predict_class)]
    logging.info(f"Predicted class: {predict_class}")


    db = SessionLocal()
    try:
        db_obj = Prediction(comment=text, predict_class=predict_class_text)
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
    except SQLAlchemyError as e:
        db.rollback()
        logging.error(f'Failed to save in DB: {e}')
    finally:
        db.close()

    return {'predicted_class': predict_class_text}


app.get("/")
async def root():
    return {
        "message": "Welcome to the Emotion Classification API. Use /predict/ to classify text."
    }


@app.get("/hello")
async def root():
    return {
        "message": "Hello, World! This is a simple FastAPI application for text classification."
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)