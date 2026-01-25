import os
import pandas as pd
import numpy as np

RAW_PATH_CSV = "data/raw/online_retail.csv"
RAW_PATH_XLSX = "data/raw/online_retail.xlsx"
OUT_PATH = "data/processed/rfm_features.csv"


def load_data() -> pd.DataFrame:
    if os.path.exists(RAW_PATH_CSV):
        df = pd.read_csv(RAW_PATH_CSV, encoding="ISO-8859-1")
        return df
    if os.path.exists(RAW_PATH_XLSX):
        df = pd.read_excel(RAW_PATH_XLSX, engine="openpyxl")
        return df
    raise FileNotFoundError(
        "Could not find dataset. Put it at data/raw/online_retail.csv or data/raw/online_retail.xlsx"
    )


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    # Strip whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    # Rename common variants to a standard schema
    rename_map = {
        "Invoice": "InvoiceNo",
        "InvoiceNo": "InvoiceNo",
        "Customer ID": "CustomerID",
        "CustomerID": "CustomerID",
        "Price": "UnitPrice",
        "UnitPrice": "UnitPrice",
        "InvoiceDate": "InvoiceDate",
        "Quantity": "Quantity",
    }

    # Apply renaming only for columns that exist
    df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})

    required_cols = {"InvoiceNo", "InvoiceDate", "CustomerID", "Quantity", "UnitPrice"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    # Remove rows with missing customer
    df = df.dropna(subset=["CustomerID"]).copy()

    # Parse dates safely
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["InvoiceDate"]).copy()

    # Remove cancellations (InvoiceNo often starts with 'C')
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df = df[~df["InvoiceNo"].str.startswith("C")].copy()

    # Remove invalid/return rows: non-positive quantity or price
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()

    # Ensure CustomerID is int-like for grouping consistency
    df["CustomerID"] = df["CustomerID"].astype(int)

    # Total spend per line item
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    return df


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    # Snapshot date: day after last transaction (common industry approach)
    snapshot_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (snapshot_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalPrice", "sum"),
        )
        .reset_index()
    )

    # Optional: guard against weird edge cases
    rfm = rfm.replace([np.inf, -np.inf], np.nan).dropna()

    # Basic sanity rules
    rfm = rfm[(rfm["Frequency"] > 0) & (rfm["Monetary"] > 0) & (rfm["Recency"] >= 0)].copy()

    return rfm


def main():
    df_raw = load_data()
    print(f"Loaded: {df_raw.shape} rows, {df_raw.shape[1]} cols")

    df_clean = clean_transactions(df_raw)
    print(f"After cleaning: {df_clean.shape} rows")

    rfm = build_rfm(df_clean)
    print(f"RFM built: {rfm.shape} customers")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    rfm.to_csv(OUT_PATH, index=False)

    print(f"Saved RFM features -> {OUT_PATH}")
    print("\nSample:\n", rfm.head())


if __name__ == "__main__":
    main()
