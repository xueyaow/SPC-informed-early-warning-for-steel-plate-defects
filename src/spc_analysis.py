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
