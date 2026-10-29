"""
Customer churn pipeline: data cleaning, EDA, model building, churn scoring.

Every function takes and returns plain pandas objects or JSON-friendly dicts, so
app.py only has to deal with files and HTTP. Charts are drawn with matplotlib's
object API (no pyplot) and returned as base64 PNG data URIs.
"""
from __future__ import annotations

import base64
import io
import re

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from pandas.api.types import is_numeric_dtype
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ----------------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
HIGH_RISK = 0.70
MEDIUM_RISK = 0.40
MIN_ROWS = 50
MAX_ROWS = 50_000
NA_VALUES = ["", "NA", "N/A", "NaN", "nan", "null", "NULL"]  # "None" is kept: it is often a real category

POSITIVE_LABELS = {
    "yes", "y", "1", "true", "t", "churn", "churned", "left",
    "exited", "attrited", "attrited customer", "cancelled", "canceled",
}
TARGET_HINTS = ("churn", "exited", "attrition", "cancel", "left")
ID_NAME = re.compile(r"^(id|(customer|cust|user|client|account|row)_?(id|number|no))$|_id$")

# Chart colours: validated categorical slots 1-4, plus status colours for risk levels.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_2, GRID = "#172033", "#52514e", "#e3e6ea"
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"
MODEL_COLORS = {
    "Logistic Regression": BLUE,
    "Decision Tree": ORANGE,
    "Random Forest": AQUA,
    "Gradient Boosting": YELLOW,
}

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": "#c9ced6",
    "axes.labelcolor": INK_2,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "legend.frameon": False,
})


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def to_py(obj):
    """Convert numpy/pandas values into plain JSON-safe Python values."""
    if isinstance(obj, dict):
        return {str(k): to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return None if (value != value or value in (float("inf"), float("-inf"))) else value
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return str(obj)


def _norm(name) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _png(fig: Figure) -> str:
    FigureCanvasAgg(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _chart(title: str, caption: str, fig: Figure) -> dict:
    return {"title": title, "caption": caption, "image": _png(fig)}


def _pct(value: float, digits: int = 1) -> str:
    return f"{value:.{digits}f}%"


def _split_columns(frame: pd.DataFrame):
    numeric = [c for c in frame.columns if is_numeric_dtype(frame[c])]
    categorical = [c for c in frame.columns if c not in numeric]
    return numeric, categorical


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def read_csv_bytes(data: bytes) -> pd.DataFrame:
    """Parse uploaded bytes into a DataFrame and run basic size checks."""
    df = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            df = pd.read_csv(
                io.BytesIO(data), encoding=encoding, low_memory=False,
                keep_default_na=False, na_values=NA_VALUES,
            )
            break
        except UnicodeDecodeError:
            continue
        except pd.errors.EmptyDataError:
            raise ValueError("The file is empty.")
        except pd.errors.ParserError:
            raise ValueError("The file could not be read as a CSV. Check that it is comma separated.")
    if df is None:
        raise ValueError("The file could not be read as a CSV.")
    if df.shape[1] < 3:
        raise ValueError("The file needs at least three columns: some features and a churn column.")
    if len(df) < MIN_ROWS:
        raise ValueError(f"The file has {len(df)} rows. At least {MIN_ROWS} rows are needed to train models.")
    if len(df) > MAX_ROWS:
        raise ValueError(f"The file has {len(df):,} rows. This tool handles up to {MAX_ROWS:,} rows.")
    return df


def read_cleaned_csv(path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False)


def binary_columns(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.columns if df[c].dropna().nunique() == 2]


def detect_target(df: pd.DataFrame) -> str | None:
    candidates = binary_columns(df)
    for column in candidates:
        if any(hint in _norm(column) for hint in TARGET_HINTS):
            return column
    return candidates[-1] if candidates else None


def encode_target(series: pd.Series):
    """Return (0/1 target, positive label, negative label)."""
    values = list(pd.unique(series.dropna()))
    if len(values) != 2:
        raise ValueError(f"The churn column must have exactly two values, but it has {len(values)}.")
    flagged = [v for v in values if str(v).strip().lower() in POSITIVE_LABELS]
    positive = flagged[0] if len(flagged) == 1 else series.value_counts().idxmin()
    negative = next(v for v in values if v != positive)
    return (series == positive).astype(int), str(positive), str(negative)


# ----------------------------------------------------------------------------
# 1. Data cleaning
# ----------------------------------------------------------------------------
def clean_dataset(raw: pd.DataFrame, target: str):
    """Return (cleaned DataFrame, id Series or None, report dict)."""
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    target = str(target or "").strip()
    if target not in df.columns:
        raise ValueError("Choose which column holds the churn label.")
    if df[target].dropna().nunique() != 2:
        raise ValueError(f"'{target}' must have exactly two values (for example Yes and No).")

    rows_before, cols_before = df.shape
    steps: list[dict] = []

    # Trim whitespace, turn blank strings into missing values
    blank_cells = 0
    for c in [c for c in df.columns if not is_numeric_dtype(df[c])]:
        trimmed = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)
        blank = trimmed.map(lambda v: isinstance(v, str) and v == "")
        blank_cells += int(blank.sum())
        df[c] = trimmed.mask(blank, np.nan)
    empty_cells = int(df.isna().sum().sum())
    steps.append({
        "title": "Trim text and mark blanks",
        "detail": f"Removed stray spaces and found {empty_cells:,} empty cells"
                  + (f" ({blank_cells:,} of them blank strings)." if blank_cells else "."),
        "changed": bool(empty_cells),
    })

    # Text columns that are really numbers (e.g. TotalCharges stored as text)
    conversions = []
    for c in [c for c in df.columns if not is_numeric_dtype(df[c]) and c != target]:
        non_null = int(df[c].notna().sum())
        if not non_null:
            continue
        converted = pd.to_numeric(df[c], errors="coerce")
        ok = int(converted.notna().sum())
        if ok / non_null >= 0.95:
            df[c] = converted
            conversions.append({"column": c, "became_missing": non_null - ok})
    steps.append({
        "title": "Convert numbers stored as text",
        "detail": ("Converted " + ", ".join(x["column"] for x in conversions) + " to numeric.")
                  if conversions else "All numeric columns already have a numeric type.",
        "changed": bool(conversions),
    })

    # Duplicates (checked before identifiers are dropped, so only exact repeats count)
    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates()
    steps.append({
        "title": "Remove duplicate rows",
        "detail": f"Removed {duplicates:,} exact duplicate rows." if duplicates else "No duplicate rows found.",
        "changed": bool(duplicates),
    })

    # Rows with no churn label cannot be used
    no_label = int(df[target].isna().sum())
    df = df.dropna(subset=[target])
    steps.append({
        "title": "Drop rows without a churn label",
        "detail": f"Dropped {no_label:,} rows." if no_label else "Every row has a churn label.",
        "changed": bool(no_label),
    })

    # Identifier and constant columns carry no signal for prediction
    dropped = []
    id_series = None
    for c in [c for c in df.columns if c != target]:
        unique = int(df[c].nunique(dropna=True))
        mostly_unique = unique >= 0.9 * len(df)
        if (ID_NAME.search(_norm(c)) and mostly_unique) or (not is_numeric_dtype(df[c]) and unique == len(df)):
            dropped.append({"column": c, "reason": "identifier, unique for every customer"})
            if id_series is None:
                id_series = df[c].copy()
        elif unique <= 1:
            dropped.append({"column": c, "reason": "same value in every row"})
    df = df.drop(columns=[d["column"] for d in dropped])
    steps.append({
        "title": "Drop identifier and constant columns",
        "detail": ("Dropped " + ", ".join(d["column"] for d in dropped) + ".") if dropped
                  else "No identifier or constant columns found.",
        "changed": bool(dropped),
    })

    # Binary 0/1 flag stored as numbers -> Yes / No
    recoded = []
    for c in [c for c in df.columns if c != target and is_numeric_dtype(df[c])]:
        if _norm(c) == "seniorcitizen" and set(df[c].dropna().unique()) <= {0, 1}:
            df[c] = df[c].map({0: "No", 1: "Yes"})
            recoded.append(c)
    steps.append({
        "title": "Recode 0/1 flags",
        "detail": ("Recoded " + ", ".join(recoded) + " from 0/1 to No/Yes.") if recoded
                  else "No 0/1 flag columns needed recoding.",
        "changed": bool(recoded),
    })

    # Merge "No internet service" / "No phone service" into plain "No"
    merge_map = {"No internet service": "No", "No phone service": "No"}
    merged = 0
    for c in [c for c in df.columns if c != target and not is_numeric_dtype(df[c])]:
        hits = int(df[c].isin(merge_map).sum())
        if hits:
            df[c] = df[c].replace(merge_map)
            merged += hits
    steps.append({
        "title": "Standardise category labels",
        "detail": f"Merged {merged:,} 'No internet service' / 'No phone service' values into 'No'." if merged
                  else "Category labels were already consistent.",
        "changed": bool(merged),
    })

    # Fill the remaining missing values
    fills = []
    for c in [c for c in df.columns if c != target]:
        missing = int(df[c].isna().sum())
        if not missing:
            continue
        if is_numeric_dtype(df[c]):
            if _norm(c) == "totalcharges" and "MonthlyCharges" in df.columns and is_numeric_dtype(df["MonthlyCharges"]):
                df[c] = df[c].fillna(df["MonthlyCharges"])
                method = "MonthlyCharges (new customers, one billing cycle)"
                if df[c].isna().any():
                    df[c] = df[c].fillna(df[c].median())
            else:
                median = float(df[c].median())
                df[c] = df[c].fillna(median)
                method = f"median ({median:.2f})"
        else:
            mode = df[c].mode(dropna=True)
            value = mode.iloc[0] if len(mode) else "Unknown"
            df[c] = df[c].fillna(value)
            method = f"most frequent value ('{value}')"
        fills.append({"column": c, "missing": missing, "method": method})
    filled_total = sum(f["missing"] for f in fills)
    steps.append({
        "title": "Fill missing values",
        "detail": f"Filled {filled_total:,} missing values in {len(fills)} column(s)." if fills
                  else "No missing values left to fill.",
        "changed": bool(fills),
    })

    df = df.reset_index(drop=True)
    if id_series is not None:
        id_series = id_series.reset_index(drop=True)
    if df.isna().any().any():
        raise ValueError("Some values are still missing after cleaning. Check the file for unusual formats.")

    numeric, categorical = _split_columns(df.drop(columns=[target]))
    report = {
        "target": target,
        "rows_before": rows_before,
        "rows_after": len(df),
        "columns_before": cols_before,
        "columns_after": df.shape[1],
        "empty_cells_found": empty_cells,
        "missing_after": 0,
        "duplicates_removed": duplicates,
        "steps": steps,
        "type_conversions": conversions,
        "dropped_columns": dropped,
        "missing_filled": fills,
        "feature_counts": {"numeric": len(numeric), "categorical": len(categorical)},
        "columns": [
            {"name": c, "kind": "target" if c == target else ("numeric" if c in numeric else "categorical"),
             "unique": int(df[c].nunique())}
            for c in df.columns
        ],
        "preview": {"columns": list(df.columns), "rows": df.head(10).values.tolist()},
        "id_column": str(id_series.name) if id_series is not None else None,
    }
    return df, id_series, to_py(report)


# ----------------------------------------------------------------------------
# 2 and 5. EDA and EDA charts
# ----------------------------------------------------------------------------
def _chart_class_balance(y: pd.Series) -> dict:
    counts = [int((y == 0).sum()), int((y == 1).sum())]
    fig = Figure(figsize=(5.2, 3.6))
    ax = fig.subplots()
    bars = ax.bar(["Retained", "Churned"], counts, color=[BLUE, ORANGE], width=0.55)
    for bar, n in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, n, f"{n:,}\n({n / len(y) * 100:.1f}%)",
                ha="center", va="bottom", color=INK, fontsize=10)
    ax.set_ylim(0, max(counts) * 1.22)
    ax.set_ylabel("Customers")
    ax.set_title("Churn distribution")
    rate = counts[1] / len(y) * 100
    return _chart("Churn distribution",
                  f"{rate:.1f}% of customers churned, so the classes are imbalanced.", fig)


def _chart_rate_by_category(feature: str, rows: list[dict], overall: float) -> dict:
    ordered = sorted(rows, key=lambda r: r["churn_rate"])
    fig = Figure(figsize=(6.4, 1.3 + 0.42 * len(ordered)))
    ax = fig.subplots()
    labels = [r["value"] for r in ordered]
    rates = [r["churn_rate"] for r in ordered]
    bars = ax.barh(labels, rates, color=BLUE, height=0.62)
    for bar, rate in zip(bars, rates):
        ax.text(rate + max(rates) * 0.015, bar.get_y() + bar.get_height() / 2, f"{rate:.1f}%",
                va="center", color=INK, fontsize=9.5)
    ax.axvline(overall, color=INK_2, linestyle="--", linewidth=1)
    ax.text(overall, len(ordered) - 0.4, f" overall {overall:.1f}%", color=INK_2, fontsize=8.5, va="bottom")
    ax.set_xlim(0, max(rates) * 1.18)
    ax.set_xlabel("Churn rate (%)")
    ax.set_title(f"Churn rate by {feature}", pad=22)
    top = ordered[-1]
    return _chart(f"Churn rate by {feature}",
                  f"'{top['value']}' customers churn most, at {top['churn_rate']:.1f}%.", fig)


def _chart_numeric(column: str, values: pd.Series, y: pd.Series) -> dict:
    fig = Figure(figsize=(6.0, 3.6))
    ax = fig.subplots()
    low, high = float(values.min()), float(values.max())
    bins = np.linspace(low, high, 31) if high > low else 10
    ax.hist(values[y == 0], bins=bins, density=True, alpha=0.62, color=BLUE, label="Retained")
    ax.hist(values[y == 1], bins=bins, density=True, alpha=0.62, color=ORANGE, label="Churned")
    ax.set_xlabel(column)
    ax.set_ylabel("Share of customers")
    ax.set_title(f"{column} by churn status")
    ax.legend(loc="upper right")
    mean_c, mean_r = float(values[y == 1].mean()), float(values[y == 0].mean())
    return _chart(f"{column} by churn status",
                  f"Churned customers average {mean_c:,.1f} {column}, retained customers {mean_r:,.1f}.", fig)


def _chart_correlation(frame: pd.DataFrame) -> dict:
    corr = frame.corr()
    names = list(corr.columns)
    cmap = LinearSegmentedColormap.from_list("blue_red", [BLUE, "#f0efec", "#e34948"])
    size = max(4.6, 0.95 * len(names) + 1.5)
    fig = Figure(figsize=(size, size * 0.86))
    ax = fig.subplots()
    image = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            value = corr.values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(value) > 0.65 else INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Correlation of numeric columns")
    fig.colorbar(image, ax=ax, shrink=0.8)
    return _chart("Correlation heatmap",
                  "Blue is a negative relationship, red a positive one. Look at the churn row first.", fig)


def run_eda(df: pd.DataFrame, target: str) -> dict:
    y, positive, negative = encode_target(df[target])
    X = df.drop(columns=[target])
    numeric, categorical = _split_columns(X)
    overall = float(y.mean() * 100)

    numeric_summary = []
    for c in numeric:
        d = X[c].describe()
        numeric_summary.append({
            "column": c, "mean": d["mean"], "std": d["std"], "min": d["min"],
            "p25": d["25%"], "median": d["50%"], "p75": d["75%"], "max": d["max"],
        })
    by_class = [{"column": c, "retained": float(X.loc[y == 0, c].mean()),
                 "churned": float(X.loc[y == 1, c].mean())} for c in numeric]

    category_rates = []
    for c in categorical:
        if not 2 <= X[c].nunique() <= 12:
            continue
        grouped = y.groupby(X[c]).agg(["mean", "size"])
        usable = grouped[grouped["size"] >= 30]
        if len(usable) >= 2:
            grouped = usable
        rows = [{"value": str(idx), "customers": int(r["size"]), "churn_rate": round(float(r["mean"] * 100), 1)}
                for idx, r in grouped.sort_values("mean", ascending=False).iterrows()]
        category_rates.append({
            "feature": c,
            "spread": round(rows[0]["churn_rate"] - rows[-1]["churn_rate"], 1),
            "rows": rows,
        })
    category_rates.sort(key=lambda d: -d["spread"])

    insights = []
    for item in category_rates[:4]:
        top, bottom = item["rows"][0], item["rows"][-1]
        insights.append(f"{item['feature']}: '{top['value']}' customers churn at {top['churn_rate']:.1f}%, "
                        f"against {bottom['churn_rate']:.1f}% for '{bottom['value']}'.")
    ranked_numeric = sorted(
        [r for r in by_class if X[r["column"]].std() > 0],
        key=lambda r: -abs(r["churned"] - r["retained"]) / X[r["column"]].std(),
    )
    for r in ranked_numeric[:2]:
        insights.append(f"{r['column']}: churned customers average {r['churned']:,.1f}, "
                        f"retained customers {r['retained']:,.1f}.")

    charts = [_chart_class_balance(y)]
    for item in category_rates[:4]:
        charts.append(_chart_rate_by_category(item["feature"], item["rows"], overall))
    for r in ranked_numeric[:3]:
        if X[r["column"]].nunique() >= 5:
            charts.append(_chart_numeric(r["column"], X[r["column"]], y))
    if numeric:
        corr_frame = X[numeric].copy()
        corr_frame[f"{target} (1 = {positive})"] = y
        charts.append(_chart_correlation(corr_frame))

    return to_py({
        "overview": {
            "rows": len(df), "features": X.shape[1], "numeric": len(numeric), "categorical": len(categorical),
            "churned": int(y.sum()), "retained": int((y == 0).sum()), "churn_rate": round(overall, 1),
            "positive_label": positive, "negative_label": negative, "target": target,
        },
        "insights": insights,
        "numeric_summary": numeric_summary,
        "numeric_by_class": by_class,
        "category_rates": category_rates[:6],
        "charts": charts,
    })


# ----------------------------------------------------------------------------
# 3, 6, 7. Model building, comparison, evaluation
# ----------------------------------------------------------------------------
def _model_catalog():
    # (name, estimator, needs scaling, description, settings)
    return [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), True,
         "Linear model on scaled numbers. Fast and easy to explain.", "max_iter = 1000"),
        ("Decision Tree", DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE), False,
         "A single tree of yes/no questions. Readable, but can overfit.", "max_depth = 6"),
        ("Random Forest", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=RANDOM_STATE), False,
         "200 trees voting together. Strong and stable on tabular data.", "n_estimators = 200, max_depth = 8"),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=RANDOM_STATE), False,
         "Trees built one after another, each correcting the last.", "scikit-learn defaults"),
    ]


def _encode_features(frame: pd.DataFrame, encoders: dict | None = None):
    """Label-encode text columns. Fits new encoders when none are supplied."""
    X = frame.copy()
    fitted = {}
    for c in [c for c in X.columns if not is_numeric_dtype(X[c])]:
        if encoders is None:
            encoder = LabelEncoder()
            X[c] = encoder.fit_transform(X[c].astype(str))
            fitted[c] = encoder
        else:
            encoder = encoders[c]
            known = set(encoder.classes_)
            values = X[c].astype(str).where(X[c].astype(str).isin(known), encoder.classes_[0])
            X[c] = encoder.transform(values)
    return X, (fitted if encoders is None else encoders)


def _importance(model, columns: list[str]) -> list[dict]:
    if hasattr(model, "feature_importances_"):
        raw = np.asarray(model.feature_importances_, dtype=float)
    else:
        raw = np.abs(np.asarray(model.coef_, dtype=float)[0])
    total = raw.sum() or 1.0
    pairs = sorted(zip(columns, raw / total), key=lambda p: -p[1])
    return [{"feature": f, "importance": float(v)} for f, v in pairs]


def _chart_roc(curves: dict, aucs: dict) -> dict:
    fig = Figure(figsize=(5.8, 4.4))
    ax = fig.subplots()
    ax.plot([0, 1], [0, 1], color="#b8bec7", linestyle="--", linewidth=1, label="Random guess")
    for name, (fpr, tpr) in curves.items():
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], linewidth=2.2, label=f"{name} (AUC {aucs[name]:.3f})")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves on the test set")
    ax.legend(loc="lower right", fontsize=9)
    return _chart("ROC curves", "Curves closer to the top-left corner separate churners from stayers better.", fig)


def _chart_confusion(cm: np.ndarray, name: str) -> dict:
    fig = Figure(figsize=(4.8, 4.0))
    ax = fig.subplots()
    ax.imshow(cm, cmap=LinearSegmentedColormap.from_list("blues", ["#eef5fd", BLUE, "#104281"]))
    labels = ["Retained", "Churned"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix: {name}")
    row_totals = cm.sum(axis=1, keepdims=True)
    for i in range(2):
        for j in range(2):
            share = cm[i, j] / row_totals[i, 0] * 100 if row_totals[i, 0] else 0
            dark = cm[i, j] > cm.max() * 0.5
            ax.text(j, i, f"{cm[i, j]:,}\n{share:.0f}% of row", ha="center", va="center",
                    color="white" if dark else INK, fontsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return _chart("Confusion matrix", "Rows are the real outcome, columns are what the model predicted.", fig)


def _chart_importance(items: list[dict], name: str) -> dict:
    top = items[:15][::-1]
    fig = Figure(figsize=(6.4, 1.2 + 0.34 * len(top)))
    ax = fig.subplots()
    bars = ax.barh([t["feature"] for t in top], [t["importance"] * 100 for t in top], color=BLUE, height=0.62)
    biggest = max(t["importance"] * 100 for t in top)
    for bar, t in zip(bars, top):
        ax.text(t["importance"] * 100 + biggest * 0.012, bar.get_y() + bar.get_height() / 2,
                f"{t['importance'] * 100:.1f}%", va="center", color=INK, fontsize=9)
    ax.set_xlim(0, biggest * 1.15)
    ax.set_xlabel("Share of total importance (%)")
    ax.set_title(f"Feature importance: {name}")
    lead = items[0]["feature"]
    return _chart("Feature importance", f"{lead} has the largest influence on the predictions.", fig)


def train_models(df: pd.DataFrame, target: str):
    """Train the four models. Returns (payload dict, bundle for scoring)."""
    y, positive, negative = encode_target(df[target])
    if y.value_counts().min() < 2:
        raise ValueError("Each class needs at least two customers to train and test a model.")
    X_raw = df.drop(columns=[target])
    X, encoders = _encode_features(X_raw)
    numeric, categorical = _split_columns(X_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    scaler = StandardScaler().fit(X_train[numeric]) if numeric else None

    def scaled(frame):
        copy = frame.copy()
        if scaler is not None:
            copy[numeric] = scaler.transform(frame[numeric])
        return copy

    rows, fitted, curves, aucs = [], {}, {}, {}
    for name, model, needs_scaling, _desc, _settings in _model_catalog():
        train_X = scaled(X_train) if needs_scaling else X_train
        test_X = scaled(X_test) if needs_scaling else X_test
        model.fit(train_X, y_train)
        preds = model.predict(test_X)
        probs = model.predict_proba(test_X)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = float(roc_auc_score(y_test, probs))
        curves[name], aucs[name], fitted[name] = (fpr, tpr), auc, (model, needs_scaling, preds)
        rows.append({
            "model": name,
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "roc_auc": auc,
        })
    rows.sort(key=lambda r: (-r["roc_auc"], -r["f1"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    best_name = rows[0]["model"]
    best_model, best_scaled, best_preds = fitted[best_name]
    cm = confusion_matrix(y_test, best_preds, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    importance = _importance(best_model, list(X.columns))
    best = rows[0]

    reading = [
        f"The test set holds {len(y_test):,} customers, {tp + fn:,} of whom actually churned.",
        f"The model caught {tp:,} of those {tp + fn:,} churners (recall {best['recall'] * 100:.1f}%) and missed {fn:,}.",
        f"It flagged {tp + fp:,} customers as likely to churn and {tp:,} of them did (precision {best['precision'] * 100:.1f}%).",
        f"It wrongly flagged {fp:,} loyal customers, so a retention offer would be wasted on them.",
    ]

    payload = {
        "model_building": {
            "target": target,
            "positive_label": positive,
            "negative_label": negative,
            "features": {"total": X.shape[1], "numeric": numeric, "categorical": categorical},
            "split": {"train": len(X_train), "test": len(X_test), "test_share": int(TEST_SIZE * 100)},
            "class_share": {"churned": round(float(y.mean() * 100), 1)},
            "steps": [
                f"Encoded {len(categorical)} text column(s) as numbers with label encoding.",
                f"Split the data {100 - int(TEST_SIZE * 100)}/{int(TEST_SIZE * 100)} into {len(X_train):,} training and "
                f"{len(X_test):,} test customers, keeping the churn ratio the same in both.",
                ("Scaled the numeric columns for Logistic Regression only." if numeric
                 else "No numeric columns to scale."),
                "Trained four models on the training customers and scored each on the unseen test customers.",
                f"Picked {best_name} because it has the highest ROC AUC ({best['roc_auc']:.3f}).",
            ],
            "models": [{"name": n, "description": d, "settings": s} for n, _m, _sc, d, s in _model_catalog()],
            "best_model": best_name,
        },
        "comparison": {
            "rows": rows,
            "best_model": best_name,
            "roc_chart": _chart_roc(curves, aucs),
        },
        "evaluation": {
            "model": best_name,
            "metrics": {k: best[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")},
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
            "reading": reading,
            "top_drivers": importance[:5],
            "confusion_chart": _chart_confusion(cm, best_name),
            "importance_chart": _chart_importance(importance, best_name),
        },
    }
    bundle = {
        "model": best_model, "name": best_name, "needs_scaling": best_scaled,
        "encoders": encoders, "scaler": scaler, "numeric": numeric,
        "columns": list(X.columns), "positive": positive, "negative": negative,
        "importance": importance,
    }
    return to_py(payload), bundle


# ----------------------------------------------------------------------------
# 4. Churn scoring and risk levels
# ----------------------------------------------------------------------------
def risk_band(probability: float) -> str:
    if probability >= HIGH_RISK:
        return "High Risk"
    if probability >= MEDIUM_RISK:
        return "Medium Risk"
    return "Low Risk"


def score_customers(df: pd.DataFrame, id_series, target: str, bundle: dict):
    """Return (scored DataFrame, summary dict)."""
    X_raw = df.drop(columns=[target])
    X, _ = _encode_features(X_raw, bundle["encoders"])
    X = X[bundle["columns"]]
    if bundle["needs_scaling"] and bundle["scaler"] is not None:
        X[bundle["numeric"]] = bundle["scaler"].transform(X[bundle["numeric"]])
    probabilities = np.round(bundle["model"].predict_proba(X)[:, 1], 4)

    scored = df.copy()
    scored["Churn_Probability"] = probabilities
    scored["Predicted_Churn"] = np.where(probabilities >= 0.5, bundle["positive"], bundle["negative"])
    scored["Risk_Category"] = [risk_band(p) for p in probabilities]
    if id_series is not None:
        scored.insert(0, str(id_series.name), id_series.values)

    levels = []
    for level, rule, colour in (
        ("High Risk", f"probability {HIGH_RISK:.2f} or higher", CRIT),
        ("Medium Risk", f"probability {MEDIUM_RISK:.2f} to {HIGH_RISK:.2f}", WARN),
        ("Low Risk", f"probability below {MEDIUM_RISK:.2f}", GOOD),
    ):
        count = int((scored["Risk_Category"] == level).sum())
        levels.append({"level": level, "rule": rule, "customers": count,
                       "share": round(count / len(scored) * 100, 1), "colour": colour})

    driver_names = [d["feature"] for d in bundle["importance"] if d["feature"] in df.columns][:4]
    id_name = str(id_series.name) if id_series is not None else None
    columns = ([id_name] if id_name else []) + driver_names + ["Churn_Probability", "Risk_Category"]
    top = scored.sort_values("Churn_Probability", ascending=False).head(15)[columns]

    fig = Figure(figsize=(6.2, 2.9))
    ax = fig.subplots()
    ordered = levels[::-1]  # Low at the bottom, High on top
    bars = ax.barh([l["level"] for l in ordered], [l["customers"] for l in ordered],
                   color=[l["colour"] for l in ordered], height=0.6)
    biggest = max(l["customers"] for l in levels) or 1
    for bar, l in zip(bars, ordered):
        ax.text(l["customers"] + biggest * 0.012, bar.get_y() + bar.get_height() / 2,
                f"{l['customers']:,} ({l['share']:.1f}%)", va="center", color=INK, fontsize=9.5)
    ax.set_xlim(0, biggest * 1.3)
    ax.set_xlabel("Customers")
    ax.set_title("Customers by risk level")
    chart = _chart("Customers by risk level",
                   f"{levels[0]['customers']:,} customers are high risk and should be contacted first.", fig)

    summary = {
        "model": bundle["name"],
        "customers": len(scored),
        "average_probability": round(float(probabilities.mean() * 100), 1),
        "predicted_churners": int((probabilities >= 0.5).sum()),
        "levels": levels,
        "thresholds": {"high": HIGH_RISK, "medium": MEDIUM_RISK},
        "top_risk": {"columns": columns, "rows": top.values.tolist()},
        "chart": chart,
        "scored_columns": list(scored.columns),
    }
    return scored, to_py(summary)
