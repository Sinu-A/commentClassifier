import logging
import time

from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logging.info("Читаю файл")
df = pd.read_csv("C:/Projects/commentsClassifier2/data/interim/data_read.csv")
df = df[["Комментарии", "Эмоциональная окраска"]]

df.columns = ["text", "label"]
print(df.head())

logging.info("Убираем лишние пробелы из окраски и приводим их в нижний регистр")
df["label"] = df["label"].apply(lambda x: x.strip().lower())
df["label"] = df["label"].map({"негативная": 0, "позитивная": 1, "нейтральная": 2, "мусор": 3})

logging.info("удаляем пропуски")
df = df.dropna()


class_values = df["label"].value_counts()

plt.figure(figsize=(7, 7))
plt.bar(
    class_values.index,
    class_values,
    label=df["label"].unique(),
)


plt.title("Комментарии классов")
plt.xlabel("Класс")
plt.ylabel("Количество")
legend_labels = ["0 - негативная", "1 - позитивная", "2 - нейтральная", "3 - мусор"]
handles = [Patch(facecolor="none", edgecolor="none", label=label) for label in legend_labels]
plt.legend(handles=handles)
plt.xticks(class_values.index)
plt.savefig("analyze.svg")


X = df["text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=42, shuffle=True
)

pd.concat([X_train, y_train], axis=1).to_csv(
    "C:/Projects/commentsClassifier2/data/processed/train.csv"
)
pd.concat([X_test, y_test], axis=1).to_csv(
    "C:/Projects/commentsClassifier2/data/processed/test.csv"
)
