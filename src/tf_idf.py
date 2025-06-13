import logging
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO)
logging.info("Читаю тест, трейн дату")
train_data = pd.read_csv("data/processed/train.csv")
test_data = pd.read_csv("data/processed/test.csv")


tfidf = TfidfVectorizer()

X_train = train_data["text"]
y_train = train_data["label"]
X_train_sparse = tfidf.fit_transform(X_train)

X_test = test_data["text"]
y_test = test_data["label"]
X_test_sparse = tfidf.transform(X_test)

logging.info("Обучаю логистическую регерссию")
logreg = LogisticRegression(max_iter=5000)

logreg.fit(X_train_sparse, y_train)
predictions = logreg.predict(X_test_sparse)

logging.info("Проверяю метрики")
print(
    f"""Метрики
        accuracy: {accuracy_score(y_test, predictions):.4f},
        precision: {precision_score(y_test, predictions, average='weighted'):.4f},
        recall: {recall_score(y_test, predictions, average='weighted'):.4f},
        f1: {f1_score(y_test, predictions, average='weighted'):.4f}"""
)

logging.info("Сохраняю модели")
filename_tfidf = "models/tfidf_preupd.sav"
filename_logreg = "models/logreg_preupd.sav"
pickle.dump((tfidf), open(filename_tfidf, "wb"))
pickle.dump((logreg), open(filename_logreg, "wb"))
