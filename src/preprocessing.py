"""Data loading, quality-check and cleaning functions."""
import pandas as pd

ORDINAL_LEVELS = {
    "Salt_Intake": ["Low", "Normal", "High", "Very High"],
    "Physical_Activity": ["Sedentary", "Light", "Moderate", "Active"],
    "Alcohol_Consumption": ["None", "Occasional", "Moderate", "Heavy"],
    "Stress_Level": ["Low", "Moderate", "High", "Very High"],
    "BP_Category": ["Normal", "Elevated", "Stage 1 Hypertension", "Stage 2 Hypertension"],
}
BINARY_MAPS = {
    "Gender": {"Female": 0, "Male": 1},
    "Family_History_HTN": {"No": 0, "Yes": 1},
    "Hypertension": {"No": 0, "Yes": 1},
}
NOMINAL_COLS = ["Smoking_Status"]

def load_data(path):
    """Read the CSV, trimming stray whitespace from text values."""
    df = pd.read_csv(path, skipinitialspace=True)
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()
    return df


# --- Data quality checks ---

def missing_summary(df):
    """Count and percentage of missing values for columns that have any."""
    missing = df.isna().sum().to_frame("n_missing")
    missing["pct_missing"] = (missing["n_missing"] / len(df) * 100).round(2)
    return missing[missing["n_missing"] > 0].sort_values("n_missing", ascending=False)


def category_counts(df):
    """Value counts (including NaN) for every text column."""
    cols = df.select_dtypes(include=["object", "string"]).columns
    return {col: df[col].value_counts(dropna=False) for col in cols}


def numeric_ranges(df):
    """Min and max of every numeric column."""
    return df.select_dtypes("number").agg(["min", "max"]).T


def acc_aha_category(sbp, dbp):
    """BP category from systolic/diastolic readings (ACC/AHA 2017 thresholds)."""
    if sbp >= 140 or dbp >= 90:
        return "Stage 2 Hypertension"
    if sbp >= 130 or dbp >= 80:
        return "Stage 1 Hypertension"
    if sbp >= 120:
        return "Elevated"
    return "Normal"


def bp_inconsistencies(df):
    """Rows with DBP >= SBP, and rows whose BP_Category disagrees with SBP/DBP."""
    inverted = df[df["Diastolic_BP_mmHg"] >= df["Systolic_BP_mmHg"]]
    derived = [acc_aha_category(s, d)
               for s, d in zip(df["Systolic_BP_mmHg"], df["Diastolic_BP_mmHg"])]
    mismatched = df[df["BP_Category"] != derived]
    return inverted, mismatched


# --- Cleaning ---

def handle_missing(df):
    """Impute missing values.

    - Alcohol_Consumption: missing means non-drinker (no "None" level exists).
    - Other categoricals: mode.
    - Numerics: median (robust to skew in the biomarkers).
    """
    df = df.copy()
    df["Alcohol_Consumption"] = df["Alcohol_Consumption"].fillna("None")
    for col in df.select_dtypes(include=["object", "string"]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    for col in df.select_dtypes("number").columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def remove_duplicates(df):
    """Drop exact duplicate rows."""
    return df.drop_duplicates().reset_index(drop=True)


def encode_categoricals(df):
    """Ordinal -> 0..n codes, binary -> 0/1, nominal -> one-hot."""
    df = df.copy()
    for col, levels in ORDINAL_LEVELS.items():
        df[col] = df[col].map({level: i for i, level in enumerate(levels)})
    for col, mapping in BINARY_MAPS.items():
        df[col] = df[col].map(mapping)
    return pd.get_dummies(df, columns=NOMINAL_COLS, dtype=int)
