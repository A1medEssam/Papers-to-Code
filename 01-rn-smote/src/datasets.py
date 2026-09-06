from pathlib import Path
import pandas as pd

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"

CORE_DATASETS = [
    "stamps", "internet_ads", "cardiotocography", "wbc",
    "lymphography", "climate", "blood", "ilpd", "qsar",
]
COMPARISON_DATASETS = ["haberman", "pima", "glass"]


def load(name):
    df = pd.read_csv(PROCESSED / f"{name}.csv")
    y = df["target"].values
    X = df.drop(columns=["target"]).values.astype(float)
    return X, y
