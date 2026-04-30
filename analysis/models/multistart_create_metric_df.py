from __future__ import annotations
import json
import re
from pathlib import Path
import pandas as pd
import argparse
import pathlib
import os

parser = argparse.ArgumentParser(description="Run two-phase integrative UDE multistart optimization.")
parser.add_argument("--phase_cut_date", type=str, required=True, help="Date to cut phases, format YYYY-MM-DD")
parser.add_argument("--town", type=str, required=True, help="Town name.")
parser.add_argument("--objective", type=str, required=False, help="Objective function to optimize", default="cases_and_conc")
parser.add_argument("--prev_phase_cut_date", type=str, required=False, help="Date to cut phases, format YYYY-MM-DD", default=None)
parser.add_argument("--n_days_pred_conc", type=int, required=False, default=0)
parser.add_argument("--substance_normalization", type=str, required=False, default="flow")
parser.add_argument("--gene_target", type=str, required=False, default="N1")

args = parser.parse_args()
town = args.town
phase_cut_date = args.phase_cut_date
prev_phase_cut_date = args.prev_phase_cut_date
objective = args.objective

if town=="Bonn":
    if not (args.substance_normalization == "flow" and args.gene_target == "N1"):
        model_path = f"{town}_{args.substance_normalization}_{args.gene_target}/multistart_models/{phase_cut_date}_{objective}"
        out_dir = f"{town}_{args.substance_normalization}_{args.gene_target}/multistart_results/{phase_cut_date}_{objective}"
    elif args.n_days_pred_conc == 0:
        model_path = f"{town}/multistart_models/{phase_cut_date}_{objective}"
        out_dir = f"{town}/multistart_results/{phase_cut_date}_{objective}"
    else:
        model_path = f"{town}/multistart_models/{phase_cut_date}_{objective}_pred_{args.n_days_pred_conc}d_conc"
        out_dir = f"{town}/multistart_results/{phase_cut_date}_{objective}_pred_{args.n_days_pred_conc}d_conc"
else:
    model_path = f"{town}/multistart_models/{phase_cut_date}_prev{prev_phase_cut_date}_{objective}"
    out_dir = f"{town}/multistart_results/{phase_cut_date}_prev{prev_phase_cut_date}_{objective}"
os.makedirs(out_dir, exist_ok=True)


def load_seed_metrics(dir_path: str | Path = model_path) -> pd.DataFrame:
    """
    Summarise all [seed]_metrics.json files under `dir_path` into one DataFrame.

    Rows: one per file/seed
    Columns: seed, all metric fields (sorted), and _file (relative path)
    """
    dir_path = Path(dir_path)
    files = sorted(dir_path.rglob("*_metrics.json"))
    if not files:
        raise FileNotFoundError(f"No *_metrics.json files found under {dir_path.resolve()}")

    records = []
    for fp in files:
        with open(fp, "r") as f:
            data = json.load(f)

        # Ensure 'seed' exists (fallback: parse from filename like '42_metrics.json')
        if "seed" not in data:
            m = re.match(r"(\d+)_metrics\.json$", fp.name)
            if m:
                data["seed"] = int(m.group(1))

        # Keep a breadcrumb to the source file
        try:
            rel = fp.relative_to(dir_path)
        except ValueError:
            rel = fp.name
        data["_file"] = str(rel)
        records.append(data)

    df = pd.DataFrame.from_records(records)

    # Arrange columns: seed | metrics (alphabetical) | _file
    metric_cols = sorted([c for c in df.columns if c not in {"seed", "_file"}])
    ordered_cols = (["seed"] if "seed" in df.columns else []) + metric_cols + (["_file"] if "_file" in df.columns else [])
    df = df[ordered_cols]

    # Coerce metric columns to numeric (safe if some are already numeric)
    if metric_cols:
        df[metric_cols] = df[metric_cols].apply(pd.to_numeric, errors="coerce")

    # Sort by seed if present
    if "seed" in df.columns:
        df = df.sort_values("seed").reset_index(drop=True)

    return df

if __name__ == "__main__":
    # Main table: one row per seed/file
    df = load_seed_metrics(model_path)
    print(df)

    
    # Highlight best (lowest) validation loss if present
    if "val_loss" in df.columns:
        best = df.nsmallest(1, "val_loss")
        cols = ["seed", "val_loss"] + (["_file"] if "_file" in df.columns else [])
        print("\nBest seed by val_loss:")
        print(best[cols])

    # Persist to disk
    df.to_csv(f"{out_dir}/multistart_metrics_{args.phase_cut_date}.csv", index=False)
