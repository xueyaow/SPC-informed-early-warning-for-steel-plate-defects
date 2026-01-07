from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


var_file = DATA_DIR / "Faults27x7_var"
data_file = DATA_DIR / "Faults.NNA"

# 1) Load column names (34 lines)
with open(var_file, "r", encoding="latin-1", errors="replace") as f:
    col_names = [line.strip() for line in f if line.strip()]

print("Column names:", len(col_names))
print("First 5 names:", col_names[:5])
print("Last 5 names:", col_names[-5:])

# 2) Load numeric data (whitespace-delimited)
df = pd.read_csv(
    data_file,
    sep=r"\s+",
    header=None,
    engine="python"
)

print("\nRaw data shape:", df.shape)

# 3) Sanity check and assign names
if df.shape[1] != len(col_names):
    raise ValueError(
        f"Column count mismatch: data has {df.shape[1]} columns, "
        f"but var file has {len(col_names)} names."
    )

df.columns = col_names

print("\nFinal dataframe shape:", df.shape)
print("First 10 columns:", df.columns[:10].tolist())
print("Last 10 columns:", df.columns[-10:].tolist())

print("\nFirst 3 rows:")
print(df.head(3).to_string(index=False))

# 4) Quick check: last 7 columns should be fault indicators (0/1)
fault_cols = df.columns[-7:]
print("\nFault columns:", fault_cols.tolist())
print("Fault value counts (sum over each fault):")
print(df[fault_cols].sum().astype(int))

import numpy as np
import matplotlib.pyplot as plt

# =========================
# Generalized SPC feature extraction
# =========================

def compute_spc_features(x, window=10):
    """
    Compute SPC-based instability features for a single CTQ signal.

    Parameters
    ----------
    x : array-like
        Sequential CTQ measurements
    window : int
        Rolling window size for local instability

    Returns
    -------
    DataFrame with SPC features
    """

    x = np.asarray(x)

    # Moving Range
    mr = np.empty_like(x, dtype=float)
    mr[:] = np.nan
    mr[1:] = np.abs(np.diff(x))

    # Estimate sigma from MR
    MR_bar = np.nanmean(mr)
    d2 = 1.128  # for n=2
    sigma_hat = MR_bar / d2

    x_bar = np.nanmean(x)
    UCL = x_bar + 3 * sigma_hat
    LCL = x_bar - 3 * sigma_hat

    df_spc = pd.DataFrame({
        "spc_violation": ((x > UCL) | (x < LCL)).astype(int),
        "moving_range": mr,
        "dist_to_UCL": UCL - x,
        "dist_to_LCL": x - LCL,
        "mr_roll_mean": pd.Series(mr).rolling(window).mean(),
        "mr_roll_std": pd.Series(mr).rolling(window).std(),
    })

    return df_spc

# =========================
# Multi-CTQ selection
# =========================

CTQ_COLS = [
    "Pixels_Areas",
    "Sum_of_Luminosity",
    "Orientation_Index",
    "Luminosity_Index",
]

print("\nSelected CTQs:")
for c in CTQ_COLS:
    print(f"{c}: {'OK' if c in df.columns else 'MISSING'}")

# =========================
# Compute SPC features for all CTQs
# =========================

spc_feature_blocks = []

for ctq in CTQ_COLS:
    features = compute_spc_features(df[ctq].values)
    features.columns = [f"{ctq}__{c}" for c in features.columns]
    spc_feature_blocks.append(features)

spc_features_all = pd.concat(spc_feature_blocks, axis=1)

print("\nSPC feature matrix shape:", spc_features_all.shape)
print("Example SPC feature columns:")
print(spc_features_all.columns[:10].tolist())

# =========================
# Merge SPC features into main dataframe
# =========================

df = pd.concat([df, spc_features_all], axis=1)

print("\nSample SPC features for Pixels_Areas:")
cols_preview = [c for c in df.columns if c.startswith("Pixels_Areas__")]
print(df[cols_preview].head(12).to_string(index=False))

# =========================
# Normalize SPC instability features
# =========================

from sklearn.preprocessing import StandardScaler

instability_cols = []

for ctq in CTQ_COLS:
    instability_cols.extend([
        f"{ctq}__spc_violation",
        f"{ctq}__moving_range",
        f"{ctq}__mr_roll_std",
    ])

instability_df = df[instability_cols].copy()

# Drop rows with NaNs for fair fusion
instability_df_clean = instability_df.dropna()

scaler = StandardScaler()
instability_scaled = scaler.fit_transform(instability_df_clean)

instability_scaled_df = pd.DataFrame(
    instability_scaled,
    columns=instability_df_clean.columns,
    index=instability_df_clean.index
)

# =========================
# Process Instability Index (PII)
# =========================

df.loc[instability_scaled_df.index, "PII"] = (
    instability_scaled_df.abs().mean(axis=1)
)

print("\nPII preview:")
print(df["PII"].dropna().head(10).to_string(index=False))

print("\nPII summary statistics:")
print(df["PII"].describe())

# =========================
# PII vs defects (Bumps)
# =========================

valid = df[["PII", "Bumps"]].dropna()
print("\nMean PII when Bumps=1:", valid.loc[valid["Bumps"] == 1, "PII"].mean())
print("Mean PII when Bumps=0:", valid.loc[valid["Bumps"] == 0, "PII"].mean())

print("\nCorrelation with Bumps:")
print("PII:",
      valid["PII"].corr(valid["Bumps"]))

pa_mr = df.loc[valid.index, "Pixels_Areas__mr_roll_std"]
print("Pixels_Areas MR std:",
      pa_mr.corr(valid["Bumps"]))

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

X_pii = valid[["PII"]]
y_pii = valid["Bumps"]

Xp_tr, Xp_te, yp_tr, yp_te = train_test_split(
    X_pii, y_pii, test_size=0.3, random_state=42, stratify=y_pii
)

pii_model = LogisticRegression(class_weight="balanced", max_iter=1000)
pii_model.fit(Xp_tr, yp_tr)

yp_pred = pii_model.predict(Xp_te)
print("\nPII-only model:")
print(classification_report(yp_te, yp_pred))

# =========================
# Compact feature set: PII + raw CTQs
# =========================

compact_features = [
    "PII",
    "Pixels_Areas",
    "Sum_of_Luminosity",
    "Orientation_Index",
    "Luminosity_Index",
]

df_compact = df[compact_features + ["Bumps"]].dropna()

Xc = df_compact[compact_features]
yc = df_compact["Bumps"]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
    Xc, yc, test_size=0.3, random_state=42, stratify=yc
)

compact_model = LogisticRegression(
    max_iter=1000, class_weight="balanced"
)
compact_model.fit(Xc_tr, yc_tr)

yc_pred = compact_model.predict(Xc_te)

print("\n=== COMPACT MODEL (PII + raw CTQs) ===")
print(classification_report(yc_te, yc_pred))
print("Confusion matrix:")
print(confusion_matrix(yc_te, yc_pred))

coef_compact = (
    pd.DataFrame({
        "feature": Xc.columns,
        "coef": compact_model.coef_[0]
    })
    .sort_values("coef", ascending=False)
)

print("\nCompact model coefficients:")
print(coef_compact.to_string(index=False))

# =========================
# Lead-time targets
# =========================

for k in [3, 5]:
    df[f"Bumps_lead_{k}"] = df["Bumps"].shift(-k)

from sklearn.metrics import classification_report

def eval_lead(k):
    lead_df = df[["PII", f"Bumps_lead_{k}"]].dropna()
    X = lead_df[["PII"]]
    y = lead_df[f"Bumps_lead_{k}"]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    m = LogisticRegression(class_weight="balanced", max_iter=1000)
    m.fit(X_tr, y_tr)
    y_pred = m.predict(X_te)

    print(f"\n=== Lead-time k={k} (PII-only) ===")
    print(classification_report(y_te, y_pred))

for k in [3, 5]:
    eval_lead(k)

# =========================
# SPC: I–MR for Pixels_Areas
# =========================

x = df["Pixels_Areas"].values

# Individuals chart
x_bar = np.mean(x)
mr = np.abs(np.diff(x))
mr_bar = np.mean(mr)

# Control chart constants for MR(2)
d2 = 1.128

sigma = mr_bar / d2
UCL_I = x_bar + 3 * sigma
LCL_I = max(0, x_bar - 3 * sigma)  # area cannot be negative

# Moving Range limits
UCL_MR = 3.267 * mr_bar
LCL_MR = 0

print("\n--- SPC Summary (Pixels_Areas) ---")
print(f"Mean: {x_bar:.2f}")
print(f"Estimated sigma: {sigma:.2f}")
print(f"I-Chart UCL: {UCL_I:.2f}")
print(f"I-Chart LCL: {LCL_I:.2f}")

# Plot I-Chart
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)
plt.figure()
plt.plot(x, marker="o", linestyle="-", markersize=2)
plt.axhline(x_bar)
plt.axhline(UCL_I)
plt.axhline(LCL_I)
plt.title("I-Chart: Pixels_Areas")
plt.xlabel("Sample index")
plt.ylabel("Pixels_Areas")
plt.savefig(FIG_DIR / "I-Chart_Pixels_Areas.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()

# Plot MR-Chart
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
FIG_DIR.mkdir(exist_ok=True)
plt.figure()
plt.plot(mr, marker="o", linestyle="-", markersize=2)
plt.axhline(mr_bar)
plt.axhline(UCL_MR)
plt.title("MR-Chart: Pixels_Areas")
plt.xlabel("Sample index")
plt.ylabel("Moving Range")
plt.savefig(FIG_DIR / "MR-Chart_Pixels_Areas.png", dpi=200, bbox_inches="tight")
plt.show()
plt.close()


# =========================
# Link SPC violations to faults
# =========================

out_of_control = (x > UCL_I) | (x < LCL_I)
df["OOC_Pixels_Area"] = out_of_control.astype(int)

print("\nOut-of-control points:", out_of_control.sum())

# Fault occurrence during OOC vs IC
fault_cols = [
    'Pastry', 'Z_Scratch', 'K_Scatch',
    'Stains', 'Dirtiness', 'Bumps', 'Other_Faults'
]

summary = {}
for f in fault_cols:
    summary[f] = {
        "faults_when_OOC": int(df.loc[df["OOC_Pixels_Area"] == 1, f].sum()),
        "faults_when_IC": int(df.loc[df["OOC_Pixels_Area"] == 0, f].sum())
    }

summary_df = pd.DataFrame(summary).T
print("\nFaults vs SPC state:")
print(summary_df)


# =========================
# SPC feature engineering
# =========================

import pandas as pd
import numpy as np

# Reuse Pixels_Areas data
x = df["Pixels_Areas"].values

# Distance to control limits
df["dist_to_UCL"] = UCL_I - x
df["dist_to_LCL"] = x - LCL_I

# Binary SPC violation flag
df["spc_violation"] = ((x > UCL_I) | (x < LCL_I)).astype(int)

# Moving range
df["moving_range"] = np.nan
df.loc[1:, "moving_range"] = np.abs(np.diff(x))

# Rolling statistics (local instability)
window = 10
df["mr_rolling_mean"] = df["moving_range"].rolling(window).mean()
df["mr_rolling_std"] = df["moving_range"].rolling(window).std()

print("\nSPC feature preview:")
print(df[[
    "Pixels_Areas",
    "spc_violation",
    "moving_range",
    "mr_rolling_mean",
    "mr_rolling_std"
]].head(15).to_string(index=False))


# =========================
# Define prediction target
# =========================

df["target_bumps"] = df["Bumps"].astype(int)

print("\nTarget distribution (Bumps):")
print(df["target_bumps"].value_counts())

feature_cols = [
    "Pixels_Areas",
    "spc_violation",
    "moving_range",
    "mr_rolling_mean",
    "mr_rolling_std",
]

X = df[feature_cols].copy()
y = df["target_bumps"].copy()

# Drop rows with NaNs (early rolling window)
mask = X.notna().all(axis=1)
X = X.loc[mask]
y = y.loc[mask]

print("\nModeling data shape:", X.shape)

# =========================
# Train logistic regression
# =========================

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nClassification report (Bumps prediction):")
print(classification_report(y_test, y_pred))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))


# =========================
# Interpret model coefficients
# =========================

# coef_df = pd.DataFrame({
#     "feature": X.columns,
#     "coefficient": model.coef_[0]
# }).sort_values(by="coefficient", ascending=False)

# print("\nLogistic regression coefficients:")
# print(coef_df)

# =========================
# Interpret model coefficients
# =========================

coef_df = pd.DataFrame({
    "feature": X.columns,
    "coefficient": model.coef_[0]
}).sort_values(by="coefficient", ascending=False)

print("\nLogistic regression coefficients (positive → higher Bumps risk):")
print(coef_df.to_string(index=False))

# =========================
# BASELINE MODEL (no SPC features)
# =========================

baseline_features = [
    "Pixels_Areas",
    "Sum_of_Luminosity",
    "Orientation_Index",
    "Luminosity_Index",
]

X_base = df[baseline_features].copy()
y_base = df["target_bumps"].copy()

# Drop NaNs (for fairness with SPC model)
mask_base = X_base.notna().all(axis=1)
X_base = X_base.loc[mask_base]
y_base = y_base.loc[mask_base]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    X_base, y_base,
    test_size=0.3,
    random_state=42,
    stratify=y_base
)

baseline_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

baseline_model.fit(Xb_train, yb_train)
yb_pred = baseline_model.predict(Xb_test)

print("\n=== BASELINE (Raw Features Only) ===")
print(classification_report(yb_test, yb_pred))
print("Confusion matrix:")
print(confusion_matrix(yb_test, yb_pred))
