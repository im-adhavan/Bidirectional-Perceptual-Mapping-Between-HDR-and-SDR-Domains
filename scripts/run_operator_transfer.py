import os
import torch
import pandas as pd
from tqdm import tqdm

from src.gpu_utils import get_device
from src.dataset_loader import stream_dataset
from src.tone_mapping import reinhard_global, filmic
from src.inverse_mapping import inverse_reinhard
from src.reconstruction_metrics import pu_error

HDR_FOLDER = r"C:\hdr_perceptual\data\hdr_exr"
RESULTS_DIR = "results"
CSV_DIR = os.path.join(RESULTS_DIR, "csv")

os.makedirs(CSV_DIR, exist_ok=True)

def main():

    device = get_device()
    print(f"Using device: {device}")

    records = []

    print("Running cross-operator transfer experiment...")

    for name, hdr in tqdm(stream_dataset(HDR_FOLDER, device=device), desc="Operator transfer"):

        ldr_train = reinhard_global(hdr)
        hdr_recon_train = inverse_reinhard(ldr_train)

        train_error = pu_error(hdr, hdr_recon_train).item()

        ldr_test = filmic(hdr)
        hdr_recon_test = inverse_reinhard(ldr_test)

        test_error = pu_error(hdr, hdr_recon_test).item()

        records.append({
            "scene": name,
            "train_operator": "reinhard",
            "test_operator": "filmic",
            "pu_error_train": train_error,
            "pu_error_transfer": test_error
        })

    df = pd.DataFrame(records)
    save_path = os.path.join(CSV_DIR, "operator_transfer_results.csv")
    df.to_csv(save_path, index=False)

    print("\n========== TRANSFER SUMMARY ==========")
    print("Mean train error:", df["pu_error_train"].mean())
    print("Mean transfer error:", df["pu_error_transfer"].mean())
    print("Saved to:", save_path)


if __name__ == "__main__":
    main()