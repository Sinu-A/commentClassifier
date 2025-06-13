import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO)
logging.info("Читаю тест, трейн дату")
train_data = pd.read_csv("data/processed/train.csv")
test_data = pd.read_csv("data/processed/test.csv")

X_train = train_data["text"]
y_train = train_data["label"]

X_test = test_data["text"]
y_test = test_data["label"]


pipeline = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=5000))])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)


param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "clf__C": [0.01, 0.1, 1, 10],
    "clf__penalty": ["l2"],
    "clf__solver": ["liblinear", "lbfgs"],
}

logging.info("Подбираю лучшие параметры grid search")
grid = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring="f1_weighted", verbose=2)
grid.fit(X_train, y_train)


logging.info("Лучшие параметры:", grid.best_params_)


y_pred = grid.predict(test_data["text"])

print("Accuracy score:", accuracy_score(test_data["label"], y_pred))
print("Precision score:", precision_score(test_data["label"], y_pred, average="weighted"))
print("Recall score:", recall_score(test_data["label"], y_pred, average="weighted"))
print("F1 score:", f1_score(test_data["label"], y_pred, average="weighted"))


logging.info("Сохраняю модели")
filename_tfidf = "models/tfidf_adv.sav"
filename_logreg = "models/grid_adv.sav"
pickle.dump((grid.best_estimator_.named_steps["tfidf"]), open(filename_tfidf, "wb"))
pickle.dump((grid.best_estimator_.named_steps["clf"]), open(filename_logreg, "wb"))
