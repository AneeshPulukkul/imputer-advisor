"""
Tutorial: Linear Regression + Feature Engineering Pipeline (scikit-learn)
=========================================================================
Topics:
  1. Linear Regression theory (in comments)
  2. Encoding (OneHot, Ordinal)
  3. Feature Scaling (StandardScaler vs MinMaxScaler)
  4. Feature Creation (custom features + PolynomialFeatures)
  5. Feature Selection (SelectKBest, VarianceThreshold)
  6. Missing-value Imputation: KNNImputer vs IterativeImputer (MICE) -- head-to-head

Run:
    python linear_regression_tutorial.py

Outputs: prints metrics + saves 3 PNG plots in current folder.
Requires: scikit-learn, pandas, numpy, matplotlib
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no GUI popup, just save PNGs
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler,
    PolynomialFeatures, FunctionTransformer,
)
from sklearn.experimental import enable_iterative_imputer  # noqa: F401 (enables MICE)
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================================
# PART 1 — WHAT IS LINEAR REGRESSION?
# ----------------------------------------------------------------------------
# Model:  y = w0 + w1*x1 + w2*x2 + ... + wp*xp + error
# Goal: choose weights w to minimize Mean Squared Error:
#         MSE = (1/n) * sum((y_true - y_pred)^2)
# sklearn solves it with SVD (no learning rate needed) in LinearRegression().
# Assumptions to remember:
#   1. Roughly linear relationship, 2. Errors have ~zero mean / constant variance,
#   3. Features aren't perfectly collinear, 4. Little extreme outlier leverage.
# Why feature engineering matters: linear models can ONLY learn linear
# boundaries in the given feature space — so encoding, scaling, creating
# interaction/polynomial features, and dropping junk features is how you
# give the model a fair chance.
# ============================================================================

def make_dataset(n_samples=1000):
    """Synthetic housing dataset with numeric + categorical cols + missingness."""
    size = np.random.normal(1500, 400, n_samples).clip(400, 3500)
    rooms = np.random.randint(1, 7, n_samples).astype(float)
    age = np.random.exponential(15, n_samples).clip(0, 80)
    dist = np.random.gamma(2.0, 2.0, n_samples)  # km to city center
    neighborhood = np.random.choice(["A-downtown", "B-suburb", "C-rural", "D-coast"],
                                    n_samples, p=[0.3, 0.4, 0.2, 0.1])
    style = np.random.choice(["apartment", "detached", "semi-detached"],
                             n_samples, p=[0.5, 0.3, 0.2])

    # True (hidden) linear relationship:
    price = (120 * size + 8000 * rooms - 900 * age - 6000 * dist
             + np.where(neighborhood == "A-downtown", 40000, 0)
             + np.where(neighborhood == "D-coast", 55000, 0)
             + np.where(style == "detached", 30000, 0)
             + np.random.normal(0, 20000, n_samples) + 50000)

    df = pd.DataFrame({
        "size_sqft": size, "num_rooms": rooms, "age_years": age,
        "dist_center_km": dist, "neighborhood": neighborhood,
        "house_style": style, "price": price,
    })
    # Inject ~10% MCAR missingness into 3 numeric cols + 1 categorical
    for col, frac in [("size_sqft", 0.08), ("age_years", 0.12),
                      ("dist_center_km", 0.10), ("house_style", 0.07)]:
        mask = np.random.rand(n_samples) < frac
        df.loc[mask, col] = np.nan
    return df


print("=" * 72)
print("PART 1: Load synthetic data")
df = make_dataset()
print(df.head(), "\n")
print("Missing values per column:\n", df.isna().sum(), "\n")
print(f"Shape: {df.shape}  | Target: 'price'\n")

X = df.drop(columns="price")
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE)

NUMERIC = ["size_sqft", "num_rooms", "age_years", "dist_center_km"]
CATEGORICAL = ["neighborhood", "house_style"]

# ============================================================================
# PART 2 — ENCODING: turn text into numbers
# ----------------------------------------------------------------------------
# - OneHotEncoder: creates 0/1 column per category. Use for NOMINAL vars
#   (neighborhood). Always set handle_unknown="ignore" so unseen test
#   categories don't crash the pipeline.
# - OrdinalEncoder: maps to 0,1,2... Use ONLY for ORDINAL vars where order
#   matters. Demo below pretends apartment < semi < detached.
# Never use LabelEncoder on features (it's for the target y only).
# ============================================================================
print("=" * 72)
print("PART 2: Encoding demo")
oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
# Note: in the real pipeline we SimpleImpute first; here fillna for demo only.
print("OneHot categories:", oh.fit(X_train[CATEGORICAL].fillna("missing")).categories_, "\n")

ord_enc = OrdinalEncoder(categories=[["apartment", "semi-detached", "detached"]])
print("Ordinal demo:",
      ord_enc.fit_transform(X_train[["house_style"]].dropna().head(3)).ravel(), "\n")

# ============================================================================
# PART 3 — SCALING: why it matters for Linear Regression
# ----------------------------------------------------------------------------
# LinearRegression itself doesn't REQUIRE scaling, but you want it when:
#  1. You compare coefficient magnitudes (unscaled: size_sqft coef looks tiny
#     vs num_rooms just because of units).
#  2. You use regularization (Ridge/Lasso) or distance-based steps
#     (KNNImputer!) — distances are dominated by the largest-scale feature.
# StandardScaler: (x - mean)/std  -> zero mean, unit variance. Default choice.
# MinMaxScaler: (x - min)/(max-min) -> [0,1]. Good for bounded/nn inputs,
#   but sensitive to outliers.
# Fit scaler on TRAIN only (inside Pipeline this is automatic).
# ============================================================================
print("=" * 72)
print("PART 3: Scaling demo (fit on train, transform both)")
scaler = StandardScaler()
tr_num = scaler.fit_transform(X_train[NUMERIC].fillna(X_train[NUMERIC].median()))
print("Means after StandardScaler (~0):", tr_num.mean(axis=0).round(3))
print("Stds  after StandardScaler (~1):", tr_num.std(axis=0).round(3), "\n")

# ============================================================================
# PART 4 — FEATURE CREATION: give a linear model non-linear power
# ----------------------------------------------------------------------------
# A) Manual/domain features: e.g. size_per_room = size/rooms, house_age_bin.
# B) PolynomialFeatures: auto-generates x1^2, x1*x2 ... LinearRegression on
#    these = polynomial regression (still linear in the WEIGHTS).
# Rule: create AFTER imputation (else NaN propagates), BEFORE scaling
# (so the scaler normalizes the new features too).
# ============================================================================
print("=" * 72)
print("PART 4: Feature creation demo")

def add_size_per_room(df_in):
    out = df_in.copy()
    out["size_per_room"] = out["size_sqft"] / out["num_rooms"].replace(0, np.nan)
    return out

creator = FunctionTransformer(add_size_per_room)  # plugs a DataFrame fn into Pipeline
print(creator.transform(X_train[NUMERIC].head(3)).round(1), "\n")

poly = PolynomialFeatures(degree=2, include_bias=False)
sample = X_train[NUMERIC].fillna(X_train[NUMERIC].median()).head(2)
print(f"PolynomialFeatures: {sample.shape[1]} cols -> {poly.fit_transform(sample).shape[1]} cols\n")

# ============================================================================
# PART 5 — FEATURE SELECTION: fewer, better features
# ----------------------------------------------------------------------------
# Why: less overfitting, faster, more interpretable coefficients.
# - VarianceThreshold: drops near-constant columns (e.g. a broken sensor).
# - SelectKBest(f_regression): keeps K features most correlated with y
#   (F-test). Simple, fast, works well before LinearRegression.
# Always select INSIDE the CV pipeline (fit on train folds only) to avoid
# leakage — done automatically when SelectKBest is a Pipeline step.
# ============================================================================
print("=" * 72)
print("PART 5: Feature selection demo")
demo_X = pd.get_dummies(X_train.fillna("missing").fillna(0), drop_first=True)
# numeric-only quick demo of SelectKBest scores:
X_n = X_train[NUMERIC].fillna(X_train[NUMERIC].median())
scores, _ = f_regression(X_n, y_train)
print(pd.Series(scores, index=NUMERIC).sort_values(ascending=False).round(1), "\n")

# ============================================================================
# PART 6 — IMPUTATION SHOWDOWN: KNN vs MICE (IterativeImputer)
# ----------------------------------------------------------------------------
# KNNImputer: fills each NaN with the mean of its K nearest complete rows
#   (distance in feature space). Simple, local. Sensitive to scale ->
#   scale first or keep units comparable. Good when similar rows exist.
# MICE / IterativeImputer: cycles through columns, regressing each missing
#   column on all others (default BayesianRidge). Global, model-based.
#   Better for correlated numeric data, slower, more hyperparameters.
# Fair comparison: two Pipelines IDENTICAL except the numeric imputer,
# evaluated with the same 5-fold CV + same test set.
# ============================================================================
print("=" * 72)
print("PART 6: KNN vs MICE — identical pipelines, only imputer differs")

cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

def make_preprocessor(imputer):
    num_pipe = Pipeline([
        ("impute", imputer),
        ("poly", PolynomialFeatures(degree=2, include_bias=False)),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("num", num_pipe, NUMERIC),
        ("cat", cat_pipe, CATEGORICAL),
    ])

def make_full_pipeline(imputer):
    return Pipeline([
        ("prep", make_preprocessor(imputer)),
        ("select", SelectKBest(f_regression, k=15)),
        ("model", LinearRegression()),
    ])

pipe_knn = make_full_pipeline(KNNImputer(n_neighbors=5))
pipe_mice = make_full_pipeline(IterativeImputer(max_iter=10, random_state=RANDOM_STATE))

scoring = {"rmse": "neg_root_mean_squared_error", "r2": "r2"}
results = {}
for name, pipe in [("KNN", pipe_knn), ("MICE", pipe_mice)]:
    cv = cross_validate(pipe, X_train, y_train, cv=5, scoring=scoring)
    results[name] = {
        "cv_rmse": -cv["test_rmse"].mean(),
        "cv_r2": cv["test_r2"].mean(),
    }
    print(f"{name}: CV RMSE={results[name]['cv_rmse']:,.0f}  CV R2={results[name]['cv_r2']:.3f}")

# Fit both on full train, score on held-out test
for name, pipe in [("KNN", pipe_knn), ("MICE", pipe_mice)]:
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results[name].update({
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "test_mae": float(mean_absolute_error(y_test, pred)),
        "test_r2": float(r2_score(y_test, pred)),
    })
    print(f"{name}: TEST RMSE={results[name]['test_rmse']:,.0f}  "
          f"MAE={results[name]['test_mae']:,.0f}  R2={results[name]['test_r2']:.3f}")

winner = min(results, key=lambda k: results[k]["test_rmse"])
print(f"\n>>> Winner on test RMSE: {winner}")
print("Rule of thumb: KNN = fast/local baseline; MICE = stronger when "
      "numerics are correlated, at the cost of fit time.\n")

# ============================================================================
# PART 7 — FINAL MODEL: inspect, diagnose, interpret
# ============================================================================
best_pipe = pipe_mice if winner == "MICE" else pipe_knn
best_pipe.fit(X_train, y_train)
y_pred = best_pipe.predict(X_test)

print("=" * 72)
print(f"PART 7: Final model ({winner}-imputed) test metrics")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):,.0f}")
print(f"MAE : {mean_absolute_error(y_test, y_pred):,.0f}")
print(f"R2  : {r2_score(y_test, y_pred):.3f}")

# Coefficient inspection (post-selection feature names need reconstructing)
try:
    prep = best_pipe.named_steps["prep"]
    sel = best_pipe.named_steps["select"]
    feat_names = prep.get_feature_names_out()
    kept = feat_names[sel.get_support()]
    coefs = best_pipe.named_steps["model"].coef_
    imp = pd.Series(coefs, index=kept).sort_values(key=np.abs, ascending=False)
    print("\nTop 10 coefficients by |magnitude| (standardized space):")
    print(imp.head(10).round(1))
except Exception as e:
    print(f"(coef inspection skipped: {e})")

# --- Plots (saved, not shown) ---
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(y_test, y_pred, alpha=0.4)
lo, hi = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
ax.plot([lo, hi], [lo, hi], "r--")
ax.set_xlabel("Actual price"); ax.set_ylabel("Predicted price")
ax.set_title(f"Actual vs Predicted ({winner})")
fig.tight_layout(); fig.savefig("plot_actual_vs_pred.png"); plt.close(fig)

fig, ax = plt.subplots()
ax.scatter(y_pred, y_test - y_pred, alpha=0.4)
ax.axhline(0, color="r", linestyle="--")
ax.set_xlabel("Predicted price"); ax.set_ylabel("Residual")
ax.set_title("Residuals vs Predicted (want: cloud around 0)")
fig.tight_layout(); fig.savefig("plot_residuals.png"); plt.close(fig)

fig, ax = plt.subplots()
ax.bar(results.keys(), [v["test_rmse"] for v in results.values()])
ax.set_ylabel("Test RMSE (lower = better)"); ax.set_title("KNN vs MICE")
fig.tight_layout(); fig.savefig("plot_knn_vs_mice.png"); plt.close(fig)

print("\nSaved: plot_actual_vs_pred.png, plot_residuals.png, plot_knn_vs_mice.png")
print("=" * 72)
print("EXERCISES:")
print(" 1. Swap StandardScaler -> MinMaxScaler. Does test RMSE change? Why/why not?")
print(" 2. Try K=5 vs K=30 in SelectKBest. Watch CV R2 for overfitting.")
print(" 3. Replace LinearRegression with Ridge(alpha=1.0). When does it win?")
print(" 4. Add RobustScaler for outlier-heavy age/dist columns and compare.")
