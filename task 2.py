import re
import sys
from pathlib import Path

import pandas as pd


def print_section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def pick_dataset_file(datasets_dir: Path) -> Path:
    supported_patterns = ("*.csv", "*.xlsx", "*.xls")
    files = []
    for pattern in supported_patterns:
        files.extend(datasets_dir.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No supported dataset file found in: {datasets_dir}\n"
            "Supported formats: .csv, .xlsx, .xls"
        )

    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0]


def load_dataset(file_path: Path) -> pd.DataFrame:
    if file_path.suffix.lower() == ".csv":
        return pd.read_csv(file_path)
    return pd.read_excel(file_path)


def print_missing_summary(df: pd.DataFrame, heading: str) -> None:
    print_section(heading)
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values found.")
    else:
        print(missing.sort_values(ascending=False))


def normalize_text(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
        return series
    text = series.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    return text.replace("", pd.NA)


def clean_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series
    converted = pd.to_numeric(
        series.astype("string").str.replace(r"[,\$\u20b9€£\s%]", "", regex=True),
        errors="coerce",
    )
    if converted.notna().mean() >= 0.6:
        return converted
    return series


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        series = df[col]
        if series.isna().sum() == 0:
            continue
        if pd.api.types.is_numeric_dtype(series):
            df[col] = series.fillna(series.median())
        elif pd.api.types.is_datetime64_any_dtype(series):
            df[col] = series.ffill().bfill()
        else:
            mode = series.mode(dropna=True)
            df[col] = series.fillna(mode.iloc[0] if not mode.empty else "Unknown")
    return df


def main() -> None:
    project_dir = Path(r"C:\Users\Hana\OneDrive\Desktop\izma internship\project 1")
    datasets_dir = Path(r"C:\Users\Hana\OneDrive\Desktop\izma internship\dataset")

    print_section("Task 2 - Data Cleaning & Preprocessing")
    print(f"Project folder: {project_dir}")
    print(f"Dataset folder: {datasets_dir}")

    dataset_file = pick_dataset_file(datasets_dir)
    print(f"Selected dataset file: {dataset_file.name}")

    df = load_dataset(dataset_file)

    print_section("Initial Dataset Snapshot")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\nData types before cleaning:")
    print(df.dtypes)

    print_missing_summary(df, "Step 1 - Missing Values (Before)")

    print_section("Step 2 - Duplicate Removal")
    dup_before = int(df.duplicated().sum())
    print(f"Duplicate rows before: {dup_before}")
    df = df.drop_duplicates().copy()
    dup_after = int(df.duplicated().sum())
    print(f"Duplicate rows after: {dup_after}")
    print(f"Rows after duplicate removal: {df.shape[0]}")

    print_section("Step 3 - Data Format Correction")
    for col in df.columns:
        lower_col = col.lower()
        series = df[col]

        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue

        series = normalize_text(series)

        if re.search(r"date|time|day|month|year", lower_col):
            series = pd.to_datetime(series, errors="coerce", dayfirst=True)
        elif re.search(r"price|discount|spend|units|amount|sales|qty|quantity", lower_col):
            series = clean_numeric(series)
        else:
            series = clean_numeric(series)

        df[col] = series

    print("Data types after cleaning:")
    print(df.dtypes)

    print_section("Step 4 - Missing Value Handling")
    df = fill_missing(df)
    print_missing_summary(df, "Missing Values (After)")

    output_file = project_dir / "task2_cleaned_dataset.csv"
    df.to_csv(output_file, index=False)

    print_section("Task 2 Completed Successfully")
    print(f"Final shape: {df.shape}")
    print(f"Cleaned dataset saved to: {output_file}")
    print("\nPreview of cleaned data (first 10 rows):")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nERROR: Task 2 failed.")
        print(f"Reason: {exc}")
        sys.exit(1)
