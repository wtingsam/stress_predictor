import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================================================================
# 1. Load CSV Data
# ================================================================
def load_data(file_path: str) -> pd.DataFrame:
    """Load questionnaire CSV file into a DataFrame."""
    print(f"[INFO] Loading questionnaire data from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"[INFO] Data loaded successfully. Shape: {df.shape}")
    return df


# ================================================================
# 2. Process Questionnaire Data
# ================================================================
def process_questionnaire(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare and engineer features from the questionnaire data."""
    print("[INFO] Processing questionnaire data...")

    # Rename columns for consistency
    df.rename(columns={df.columns[0]: 'date', df.columns[1]: 'id'}, inplace=True)

    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Compute total score as sum of all questionnaire values
    df['total_score'] = df.iloc[:, 3:].sum(axis=1)

    # Create binary stress index
    df['stress_index'] = np.where(df['total_score'] > 20, 1, 0)

    # Create categorical stress level
    def get_stress_level(score):
        if 0 < score <= 13:
            return 'low'
        elif 13 < score <= 26:
            return 'medium'
        else:
            return 'high'

    df['stress_level'] = df['total_score'].apply(get_stress_level)

    print("[INFO] Questionnaire processing complete.")
    print(df[['date', 'id', 'total_score', 'stress_index', 'stress_level']].head())
    return df


# ================================================================
# 3. Process Ring Data
# ================================================================
def process_ring_data(ringdata_path: str, ids: list) -> pd.DataFrame:
    """Aggregate ring data JSON files into a summary DataFrame."""
    print(f"[INFO] Processing ring data from: {ringdata_path}")

    data_all = []

    for user_id in ids:
        user_folder = os.path.join(ringdata_path, str(user_id))
        if not os.path.isdir(user_folder):
            print(f"[WARN] No folder found for ID: {user_id}")
            continue

        print(f"[INFO] Loading ring data for ID: {user_id}")
        json_files = [f for f in os.listdir(user_folder) if f.endswith('.json')]

        for json_file in json_files:
            file_path = os.path.join(user_folder, json_file)

            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Skip files with errors
                if data.get("error") is not None:
                    print(f"[WARN] Skipping {json_file}: error in file")
                    continue

                record = {
                    "id": user_id,
                    "file_name": os.path.splitext(json_file)[0]
                }
                record["date_ringdata"] = record["file_name"].split("_")[1]
                record["date_ringdata"] = pd.to_datetime(record["date_ringdata"])

                # Process metrics
                for metric in data.get("data", {}).get("metric_data", []):
                    metric_type = metric.get("type")
                    if metric_type not in ['hr', 'temp', 'hrv', 'steps']:
                        continue

                    df_metric = pd.DataFrame(metric["object"]["values"])
                    col_suffix = f"_{metric_type}"

                    record[f"mean{col_suffix}"] = round(df_metric["value"].mean(), 2)
                    record[f"std{col_suffix}"] = round(df_metric["value"].std(), 2)
                    record[f"min{col_suffix}"] = round(df_metric["value"].min(), 2)
                    record[f"max{col_suffix}"] = round(df_metric["value"].max(), 2)

                    # Moving averages (2–5)
                    for window in range(2, 6):
                        ma_col = f"ma{window}{col_suffix}"
                        df_metric[ma_col] = df_metric["value"].rolling(window=window).mean()
                        record[ma_col] = round(df_metric[ma_col].mean(), 2)

                    # Mean within 90th percentile
                    p90 = df_metric["value"].quantile(0.9)
                    record[f"mean_90th_percentile{col_suffix}"] = round(
                        df_metric[df_metric["value"] <= p90]["value"].mean(), 2
                    )

                data_all.append(record)

            except Exception as e:
                print(f"[ERROR] Failed to load {json_file} for ID {user_id}: {e}")

    df_all = pd.DataFrame(data_all)
    print(f"[INFO] Ring data summary created. Shape: {df_all.shape}")
    return df_all


# ================================================================
# 4. Merge Questionnaire and Ring Data
# ================================================================
def merge_datasets(df_score: pd.DataFrame, df_ring: pd.DataFrame) -> pd.DataFrame:
    """Merge questionnaire and ring data on ID and date."""
    print("[INFO] Merging questionnaire and ring data...")

    df_merged = pd.merge(
        df_score[['date', 'id', 'total_score', 'stress_index', 'stress_level']],
        df_ring,
        left_on=['id', 'date'],
        right_on=['id', 'date_ringdata'],
        how='inner'
    )

    print(f"[INFO] Merge complete. Shape: {df_merged.shape}")
    return df_merged


# ================================================================
# 5. Add Lag Features
# ================================================================
def add_lag_features(df: pd.DataFrame, feature_columns: list, max_lag: int = 5) -> pd.DataFrame:
    """Add lag features grouped by 'id'."""
    print(f"[INFO] Adding lag features (max_lag={max_lag})...")
    lagged_frames = []

    for lag in range(1, max_lag + 1):
        lagged = df.groupby('id')[feature_columns].shift(lag)
        lagged.columns = [f"{col}_lag{lag}" for col in feature_columns]
        lagged_frames.append(lagged)

    df_lagged = pd.concat([df] + lagged_frames, axis=1)
    print(f"[INFO] Lag features added. Total new columns: {len(lagged_frames) * len(feature_columns)}")
    return df_lagged


# ================================================================
# 6. Save Outputs
# ================================================================
def save_outputs(df_merged: pd.DataFrame):
    """Save merged dataset and latest snapshot per ID."""
    os.makedirs("data", exist_ok=True)

    merged_path = "data/merged_score_ringdata.csv"
    latest_path = "data/merged_score_ringdata_latest.csv"

    # Save full merged dataset
    df_merged.to_csv(merged_path, index=False)
    print(f"[INFO] Full merged dataset saved → {merged_path}")

    # Save latest data per ID
    df_latest = df_merged.sort_values(by=["id", "date"], ascending=[True, False])
    df_latest = df_latest.groupby("id").head(1).reset_index(drop=True)
    df_latest.to_csv(latest_path, index=False)
    print(f"[INFO] Latest data per ID saved → {latest_path}")


# ================================================================
# 7. Main Pipeline
# ================================================================
if __name__ == "__main__":
    try:
        file_path = "data/smartring_pilot_questionnaire.csv"
        ringdata_path = "data/smartring_pilot_ringdata"

        # Load and process data
        df_score = process_questionnaire(load_data(file_path))
        df_ring = process_ring_data(ringdata_path, df_score["id"].unique())

        # Merge datasets
        df_merged = merge_datasets(df_score, df_ring)

        # Count number of data points per ID
        data_points_per_id = df_merged.groupby('id').size().reset_index(name='num_data_points')

        print("[INFO] Number of data points per ID:")
        print(data_points_per_id)

        # Select numeric columns for lag features and exclude total_score, stress_index, stress_level from lag features
        numeric_cols = [
            col for col in df_merged.select_dtypes(include=[np.number]).columns
            if col not in ['total_score', 'stress_index', 'stress_level']
        ]
        # Sort by id and date before lagging
        df_merged.sort_values(by=['id', 'date'], inplace=True)
        
        # Add lag features
        df_merged = add_lag_features(df_merged, numeric_cols, max_lag=5)

        # Fill NaNs resulting from lagging
        df_merged.fillna(method='bfill', inplace=True)

        # Save outputs
        save_outputs(df_merged)

        print("[SUCCESS] All processing completed successfully!")

    except Exception as e:
        print(f"[FATAL] Script failed: {e}")
