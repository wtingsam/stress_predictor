import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, recall_score, confusion_matrix,
    mean_squared_error, roc_auc_score
)
from scipy.stats import pearsonr


# ================================================================
# 1. LOAD DATA
# ================================================================
def load_data(file_path):
    """Load dataset from a CSV file."""
    print(f"[INFO] Loading data from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"[INFO] Data loaded successfully with shape: {df.shape}")
    return df


# ================================================================
# 2. TRAIN/VALIDATION SPLIT
# ================================================================
def split_train_validation(df, random_seed):
    """Split dataset by IDs into train and validation groups."""
    ids = df['id'].unique()
    np.random.seed(random_seed)
    validation_ids = np.random.choice(ids, size=min(5, len(ids)), replace=False)

    df_validation = df[df['id'].isin(validation_ids)].groupby('id').tail(1).reset_index(drop=True)
    df_train = df[~df['id'].isin(validation_ids)].groupby('id').tail(1).reset_index(drop=True)

    print(f"[INFO] Split complete — Train: {df_train.shape}, Validation: {df_validation.shape}")
    return df_train, df_validation


def monte_carlo_simulation(df, feature_columns, model_class, target, task, num_runs, debug=False):
    """Run multiple random train/validation splits, collect metrics and feature importances."""
    print(f"\n[INFO] Starting Monte Carlo simulation with {num_runs} runs...\n")

    results = []
    all_true, all_pred, all_pred_proba = [], [], []
    all_importances = []

    from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay

    for run in range(1, num_runs + 1):
        print(f"\n========== Monte Carlo Run {run}/{num_runs} ==========")
        random_seed = np.random.randint(0, 10000)

        # Ensure both classes in binary task
        max_attempts = 20
        attempt = 0
        while attempt < max_attempts:
            df_train, df_val = split_train_validation(df, random_seed)
            y_train = df_train[target]
            if task == 'regression' or (len(np.unique(y_train)) > 1):
                break
            random_seed = np.random.randint(0, 10000)
            attempt += 1
        if len(np.unique(y_train)) < 2 and task != 'regression':
            print(f"[WARN] Skipping run {run} — only one class in training data.")
            continue

        X_train, y_train = df_train[feature_columns], df_train[target]
        X_val, y_val = df_val[feature_columns], df_val[target]

        # Train model
        model = model_class(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=3,
            random_state=random_seed
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_true = y_val.values

        # Metrics
        metrics = {}
        if task == 'regression':
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['r2'] = model.score(X_val, y_val)
        else:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['sensitivity_mean'] = np.mean(
                recall_score(y_true, y_pred, average=None, zero_division=0)
            )

            # Try ROC-AUC
            try:
                y_proba = model.predict_proba(X_val)[:, 1]
                auc_val = roc_auc_score(y_true, y_proba)
                metrics['auc'] = auc_val
                all_pred_proba.extend(y_proba)
            except Exception as e:
                print(f"[WARN] ROC-AUC computation failed: {e}")
                metrics['auc'] = np.nan

        metrics['run'] = run
        results.append(metrics)

        # Accumulate predictions and true values for visualization
        all_true.extend(y_true)
        all_pred.extend(y_pred)

        # Store feature importances
        if hasattr(model, "feature_importances_"):
            importance_df = pd.DataFrame({
                'feature': feature_columns,
                f'importance_run{run}': model.feature_importances_
            })
            all_importances.append(importance_df)

        if debug:
            print(f"[DEBUG] Run {run} Metrics: {metrics}")
            exit(0)

    # Combine metrics
    results_df = pd.DataFrame(results)
    print("\n[INFO] Monte Carlo Results Summary:")
    print(results_df.describe())

    os.makedirs("results", exist_ok=True)

    # ================================================================
    # 4A. REGRESSION SCATTER PLOT
    # ================================================================
    if task == 'regression' and all_true and all_pred:
        corr, p_value = pearsonr(all_true, all_pred)
        r2_all = np.corrcoef(all_true, all_pred)[0, 1] ** 2
        print(f"[INFO] Pearson correlation: {corr:.3f} (p={p_value:.3e})")
        print(f"[INFO] Overall R²: {r2_all:.3f}")

        plt.figure(figsize=(6, 6))
        plt.scatter(all_true, all_pred, alpha=0.6, color='dodgerblue', edgecolor='k')
        plt.plot([min(all_true), max(all_true)], [min(all_true), max(all_true)], 'r--', lw=2)
        plt.legend([f"r = {corr:.3f}, R² = {r2_all:.3f}"], loc='upper left')
        plt.xlabel("True Total Score")
        plt.ylabel("Predicted Total Score")
        plt.title("True vs Predicted Total Score (All Runs)")
        plt.grid(True, linestyle='--', alpha=0.5)

        scatter_path = os.path.join("results", "true_vs_predicted_total_score_all_runs.png")
        plt.savefig(scatter_path, dpi=300)
        plt.close()
        print(f"[INFO] Combined scatter plot saved → {scatter_path}")

    # ================================================================
    # 4B. BINARY CLASSIFICATION PERFORMANCE PLOTS
    # ================================================================
    elif task == 'binary' and all_true and all_pred:
        from sklearn.metrics import RocCurveDisplay

        # --- Confusion Matrix ---
        cm = confusion_matrix(all_true, all_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', values_format='d')
        plt.title("Confusion Matrix (All Runs)")
        cm_path = os.path.join("results", "confusion_matrix_all_runs.png")
        plt.savefig(cm_path, dpi=300)
        plt.close()
        print(f"[INFO] Confusion matrix plot saved → {cm_path}")

        # --- ROC Curve ---
        if all_pred_proba:
            fpr, tpr, _ = roc_curve(all_true, all_pred_proba)
            roc_auc = auc(fpr, tpr)

            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, color='dodgerblue', lw=2, label=f"AUC = {roc_auc:.3f}")
            plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate (Sensitivity)")
            plt.title("ROC Curve (All Monte Carlo Runs)")
            plt.legend(loc="lower right")
            plt.grid(True, linestyle='--', alpha=0.6)

            roc_path = os.path.join("results", "roc_curve_all_runs.png")
            plt.savefig(roc_path, dpi=300)
            plt.close()
            print(f"[INFO] ROC curve plot saved → {roc_path}")

    # ================================================================
    # 5. FEATURE IMPORTANCE AGGREGATION & PLOT
    # ================================================================
    if all_importances:
        print("\n[INFO] Aggregating feature importances across runs...")
        importance_merged = all_importances[0]
        for imp_df in all_importances[1:]:
            importance_merged = importance_merged.merge(imp_df, on="feature", how="outer")

        importance_merged.fillna(0, inplace=True)
        importance_merged["avg_importance"] = importance_merged.drop(columns=["feature"]).mean(axis=1)

        top20 = importance_merged.sort_values("avg_importance", ascending=False).head(20)
        plt.figure(figsize=(10, 6))
        plt.barh(top20["feature"][::-1], top20["avg_importance"][::-1],
                 color='skyblue', edgecolor='k')
        plt.xlabel("Average Feature Importance")
        plt.title("Top 20 Average Feature Importances (Across Monte Carlo Runs)")
        plt.tight_layout()

        importance_plot_path = os.path.join("results", "top20_average_feature_importances.png")
        plt.savefig(importance_plot_path, dpi=300)
        plt.close()
        print(f"[INFO] Top-20 average feature importance plot saved → {importance_plot_path}")

        importance_csv_path = os.path.join("results", "average_feature_importances.csv")
        importance_merged.to_csv(importance_csv_path, index=False)
        print(f"[INFO] Average feature importances saved → {importance_csv_path}")

    return results_df



# ================================================================
# 6. MAIN PIPELINE
# ================================================================
def main(args):
    df = load_data(args.data_path)
    df.sort_values(by=['id', 'date'], inplace=True)

    feature_columns = [
        col for col in df.columns
        if col not in ['date', 'id', 'total_score', 'stress_index',
                       'stress_level', 'file_name', 'date_ringdata']
    ]

    df['stress_level'] = df['stress_level'].map({'low': 0, 'medium': 1, 'high': 2})

    # Define model & target
    if args.task == 'regression':
        target, model_class = 'total_score', GradientBoostingRegressor
    elif args.task == 'binary':
        target, model_class = 'stress_index', GradientBoostingClassifier
    elif args.task == 'classification':
        target, model_class = 'stress_level', GradientBoostingClassifier
    else:
        raise ValueError("Invalid task type. Choose regression, binary, or classification.")

    results_df = monte_carlo_simulation(
        df, feature_columns, model_class, target, args.task, args.num_runs, debug=args.debug
    )

    os.makedirs("results", exist_ok=True)
    results_path = os.path.join("results", f"monte_carlo_results_{args.task}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"[INFO] Monte Carlo results saved to {results_path}")


# ================================================================
# 7. ENTRY POINT
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo Training for Gradient Boosting Models")
    parser.add_argument('--task', choices=['regression', 'binary', 'classification'], required=True)
    parser.add_argument('--data_path', type=str, default='data/merged_score_ringdata.csv')
    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    main(args)
