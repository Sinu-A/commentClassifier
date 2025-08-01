import logging
import pickle
import torch

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from transformers import BertTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification
from transformers import BertConfig, AutoTokenizer
from safetensors.torch import load_file



from database_bert import Prediction, SessionLocal, init_db


logging.basicConfig(level=logging.INFO)
logging.info("Инициализирую приложение FastApi")


tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruRoberta-large")
model = AutoModelForSequenceClassification.from_pretrained("ai-forever/ruRoberta-large", num_labels=4)
BERT_PATH = load_file("models/model.safetensors")
model.load_state_dict(BERT_PATH, strict=False)
model.eval()


ID2TEXT = {0: "негативная", 1: "позитивная", 2: "нейтральная", 3: "мусор"}

app = FastAPI()


@app.on_event("startup")
async def on_startup():
    init_db()



class TextRequest(BaseModel):
    text: str


@app.post("/predict/")
async def predict(request: TextRequest):
    logging.info(f"Recieved text for prediction: {request.text}")
    text = request.text

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
        predict_class_text = ID2TEXT[predicted_class_id]

    logging.info(f"Predicted class: {predicted_class_id} ({predict_class_text})")

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