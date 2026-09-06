import numpy as np
import pandas as pd

RNG = np.random.RandomState(42)
RAW = "/home/a1medessam/Desktop/papers2code/rnsmote_project/data/raw"
OUT = "/home/a1medessam/Desktop/papers2code/rnsmote_project/data/processed"

PAPER_TABLE1 = {
    "stamps":          dict(minors=6,   majors=309, features=11),
    "internet_ads":    dict(minors=32,  majors=1598, features=1557),
    "cardiotocography":dict(minors=33,  majors=1648, features=23),
    "wbc":             dict(minors=10,  majors=444, features=11),
    "lymphography":    dict(minors=6,   majors=142, features=20),
    "climate":         dict(minors=46,  majors=494, features=21),
    "blood":           dict(minors=178, majors=570, features=5),
    "ilpd":            dict(minors=167, majors=416, features=11),
    "qsar":            dict(minors=356, majors=699, features=42),
}

results = {}

def save(name, X, y):
    """y must be 0/1 with 1 = minority, matching the paper's convention."""
    df = pd.DataFrame(X)
    df["target"] = y
    df.to_csv(f"{OUT}/{name}.csv", index=False)
    n_minor = int((y == 1).sum())
    n_major = int((y == 0).sum())
    results[name] = dict(minors=n_minor, majors=n_major, features=X.shape[1])

# --- 1 & 2: Stamps & Internet Ads -----------------------------------------
# Verified superset (same majority pool, larger minority pool) from a DAMI
# mirror; subsample the minority class down to the paper's exact count.
stamps = np.load(f"/home/a1medessam/Desktop/papers2code/tmp_outlierdet/formatted_data/stamps.npz")
X, y = stamps["X"], stamps["y"]
minor_idx = np.where(y == 1)[0]
major_idx = np.where(y == 0)[0]
keep_minor = RNG.choice(minor_idx, size=6, replace=False)
keep = np.concatenate([major_idx, keep_minor])
save("stamps", X[keep], y[keep].astype(int))

ads = np.load(f"/home/a1medessam/Desktop/papers2code/tmp_outlierdet/formatted_data/internetads.npz")
X, y = ads["X"], ads["y"]
minor_idx = np.where(y == 1)[0]
major_idx = np.where(y == 0)[0]
keep_minor = RNG.choice(minor_idx, size=32, replace=False)
keep = np.concatenate([major_idx, keep_minor])
save("internet_ads", X[keep], y[keep].astype(int))

# --- 3: Cardiotocography ----------------------------------------------------
df = pd.read_csv(f"/home/a1medessam/Desktop/papers2code/Outlier-detection/Numerical/original/cardiotocography_2and3_33_variant1ori.csv", header=None)
y = (df.iloc[:, -1] == "yes").astype(int).values
X = df.iloc[:, :-1].values
save("cardiotocography", X, y)

# --- 4: WBC ------------------------------------------------------------------
df = pd.read_csv(f"/home/a1medessam/Desktop/papers2code/Outlier-detection/Numerical/original/wbc_malignant_39_variant1ori.csv", header=None)
y_full = (df.iloc[:, -1] == 2).astype(int).values
X_full = df.iloc[:, :-1].values
minor_idx = np.where(y_full == 1)[0]
major_idx = np.where(y_full == 0)[0]
keep_minor = RNG.choice(minor_idx, size=10, replace=False)
keep = np.concatenate([major_idx, keep_minor])
save("wbc", X_full[keep], y_full[keep])

# --- 5: Lymphography ---------------------------------------------------------
df = pd.read_csv(f"/home/a1medessam/Desktop/papers2code/Outlier-detection/Nominal/original/lymphographyori.csv", header=None)
df = df[df.iloc[:, -1].isin(["1", "2", "3", "4"])]  # drop malformed row(s)
label = df.iloc[:, -1].astype(int)
y = label.isin([1, 4]).astype(int).values           # rare classes = minority/outlier
X = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce").values
save("lymphography", X, y)

# --- 6: Climate model simulation crashes -------------------------------------
df = pd.read_csv("/home/a1medessam/Desktop/papers2code/tmp_climate/pop_failures.dat", sep=r"\s+", engine="python")
y = (df["outcome"] == 0).astype(int).values  # 0 = simulation failure = minority
X = df.drop(columns=["outcome"]).values
save("climate", X, y)

# --- 7: Blood transfusion -----------------------------------------------------
df = pd.read_csv("/home/a1medessam/Desktop/papers2code/tmp_blood/transfusion.data.txt")
y = df.iloc[:, -1].astype(int).values
X = df.iloc[:, :-1].values
save("blood", X, y)

# --- 8: ILPD --------------------------------------------------------------------
df = pd.read_csv("/home/a1medessam/Desktop/papers2code/mikeizbicki_ds/csv/uci/Indian Liver Patient Dataset (ILPD).csv", header=None)
df[9] = df[9].fillna(df[9].median())  # 4 rows have a missing A/G ratio; impute rather than drop, to preserve all 583 records like the paper's reported count
gender_col = 1
df[gender_col] = (df[gender_col] == "Female").astype(int)
y = (df.iloc[:, -1] == 2).astype(int).values  # 2 = non-patient = minority per Table 1
X = df.iloc[:, :-1].values
save("ilpd", X, y)

# --- 9: QSAR biodegradation ------------------------------------------------------
df = pd.read_csv("/home/a1medessam/Desktop/papers2code/tmp_QSAR-Biodegradation/NewBioDeg.csv", header=None)
y = (df.iloc[:, -1] == "RB").astype(int).values
X = df.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
n_missing = int(X.isna().sum().sum())
X = X.fillna(X.median()).values  # this mirror has a small number of parsing
# gaps (~1% of cells) not present in the original UCI release; median-imputed
# rather than dropping ~25% of rows, since every affected row has only 1-2 gaps
save("qsar", X, y)

# --- comparison trio for the SMOTE-LOF benchmark (paper section 4.6) -------------
haberman = pd.read_csv(f"{RAW}/haberman.csv", header=None)
save("haberman", haberman.iloc[:, :-1].values, (haberman.iloc[:, -1] == 2).astype(int).values)

pima = pd.read_csv(f"{RAW}/pima-indians-diabetes.csv", header=None)
save("pima", pima.iloc[:, :-1].values, pima.iloc[:, -1].astype(int).values)

glass = pd.read_csv(f"{RAW}/glass.csv", header=None)
rarest_class = glass.iloc[:, -1].value_counts().idxmin()
save("glass", glass.iloc[:, :-1].values, (glass.iloc[:, -1] == rarest_class).astype(int).values)

# --- verification report ---------------------------------------------------------
print(f"{'dataset':<18}{'minors':>8}{'majors':>8}{'features':>10}   vs paper")
for name, spec in PAPER_TABLE1.items():
    r = results[name]
    match = "EXACT" if (r["minors"], r["majors"]) == (spec["minors"], spec["majors"]) else "DIFFERS"
    print(f"{name:<18}{r['minors']:>8}{r['majors']:>8}{r['features']:>10}   "
          f"paper=({spec['minors']},{spec['majors']},{spec['features']}f)  {match}")
print()
for name in ["haberman", "pima", "glass"]:
    r = results[name]
    print(f"{name:<18}{r['minors']:>8}{r['majors']:>8}{r['features']:>10}   (comparison set)")
