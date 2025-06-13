import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logging.info("Start reading files")

main_path = Path("./data/raw/data_raw.xlsx")
sheet_names = ["UX", "GP", "AS"]

df = []
for sheet_name in tqdm(sheet_names, desc="Читаем листы"):
    data = pd.read_excel(main_path, sheet_name=sheet_name)[
        ["Комментарии", "Эмоциональная окраска"]
    ]
    df.append(data)

df = pd.concat(df)

df.to_csv("./data/processed/data_read.csv")

logging.info("Done")
