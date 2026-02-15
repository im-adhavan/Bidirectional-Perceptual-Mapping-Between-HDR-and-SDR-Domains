import os
import torch
import pandas as pd
from tqdm import tqdm

from src.gpu_utils import get_device
from src.dataset_loader import stream_dataset

from src.tone_mapping import reinhard_global
from src.inverse_mapping import inverse_reinhard

from src.scene_features import extract_scene_features
from src.reconstruction_metrics import rmse, pu_error

from src.regression_analysis import run_multivariate_regression
from src.display_analysis import evaluate_display_peaks
from src.causal_analysis import highlight_causality
from src.stability_ranking import rank_operators

from src.plotting import (
    plot_correlation_matrix,
    plot_error_distribution,
    plot_error_vs_feature,
)


HDR_FOLDER = r"C:\hdr_perceptual\data\hdr_exr"

RESULTS_DIR = "results"
CSV_DIR = os.path.join(RESULTS_DIR, "csv")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


def main():

    device = get_device()
    print(f"Using device: {device}")

    display_peaks = evaluate_display_peaks([100, 400, 1000])

    records = []

    print("\nStreaming HDR ↔ SDR Evaluation...")

    for name, hdr in tqdm(
        stream_dataset(HDR_FOLDER, device=device),
        desc="Streaming EXR scenes",
    ):

        features = extract_scene_features(hdr)

        for peak in display_peaks:

            ldr = reinhard_global(hdr)

            hdr_recon = inverse_reinhard(ldr)

            rec = {
                "scene": name,
                "display_peak": peak,
                "rmse": rmse(hdr, hdr_recon).item(),
                "pu_error": pu_error(hdr, hdr_recon, peak=peak).item(),
            }

            rec.update(features)
            records.append(rec)

        del hdr, ldr, hdr_recon
        torch.cuda.empty_cache()


    df = pd.DataFrame(records)

    metrics_path = os.path.join(CSV_DIR, "bidirectional_metrics.csv")
    df.to_csv(metrics_path, index=False)

    print(f"\nSaved metrics to: {metrics_path}")


    print("\nRunning multivariate regression...")

    feature_columns = [
        "dynamic_range",
        "log_std",
        "highlight_ratio",
        "shadow_ratio",
    ]

    regression_results = run_multivariate_regression(
        df=df,
        feature_columns=feature_columns,
        target_column="pu_error",
        polynomial_degree=2,
        use_ridge=True,
        ridge_alpha=1.0,
        k_folds=5,
    )

    reg_path = os.path.join(CSV_DIR, "regression_summary.csv")

    reg_df = pd.DataFrame({
        "r2": [regression_results["r2"]],
        "cv_r2": [regression_results["cv_r2"]],
    })

    reg_df.to_csv(reg_path, index=False)

    print(f"Saved regression summary to: {reg_path}")


    print("\nFeature correlations with PU error:")

    for col in feature_columns:
        corr = df[col].corr(df["pu_error"])
        print(f"{col}: {corr:.4f}")


    highlight_corr = highlight_causality(df)
    print(f"\nHighlight → PU error correlation: {highlight_corr:.4f}")


    df["operator"] = "reinhard"
    ranking = rank_operators(df)

    ranking_path = os.path.join(CSV_DIR, "operator_ranking.csv")
    ranking.to_csv(ranking_path)

    print(f"Operator ranking saved to: {ranking_path}")


    print("\nGenerating diagnostic plots...")

    plot_correlation_matrix(
        df[feature_columns + ["pu_error"]],
        os.path.join(FIG_DIR, "correlation_matrix.png"),
    )

    plot_error_distribution(
        df,
        error_col="pu_error",
        save_path=os.path.join(FIG_DIR, "pu_error_distribution.png"),
    )

    for feature in feature_columns:
        plot_error_vs_feature(
            df,
            feature_col=feature,
            error_col="pu_error",
            save_path=os.path.join(
                FIG_DIR,
                f"pu_error_vs_{feature}.png",
            ),
        )

    print("\n========== FINAL SUMMARY ==========")
    print("R²:", regression_results["r2"])
    print("CV R²:", regression_results["cv_r2"])
    print("All outputs saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()