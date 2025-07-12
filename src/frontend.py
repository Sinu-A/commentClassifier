import logging

import requests
import streamlit as st


logging.basicConfig(level=logging.INFO)
logging.info("Launching Streamlit apps")

st.title("Классификация текста")

text = st.text_area("Введите текст для классификации:")

if st.button("Классифицировать"):
    if text.strip():
        logging.info(f"Получен текст для классификации: {text}")
        logging.info("Отправка запроса на сервер")
        response = requests.post("http://localhost:8000/predict/", json={"text": text})
        if response.ok:
            logging.info("Получен ответ от сервера")
            result = response.json()
            st.success(f"Класс: {result['predicted_class']}")
        else:
            logging.error(f"Ошибка при обращении к серверу: {response.status_code}")
            st.error("Ошибка при обращении к серверу")
    else:
        st.warning("Пожалуйста, введите текст")