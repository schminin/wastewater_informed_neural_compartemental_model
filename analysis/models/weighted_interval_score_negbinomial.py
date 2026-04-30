import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import nbinom

sys.path.append(str(Path(__file__).resolve().parents[2]))
from optimization import optimization_utils


WIS_ALPHAS = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=float)


def mean_vmr_to_nbinom_params(mu, vmr, eps=1e-8):
    p = np.clip(1.0 / vmr, eps, 1.0 - eps)
    r = np.clip(mu / np.maximum(vmr - 1.0, eps), eps, 1e12)
    return r, p


def nbinom_mixture_cdf(k, r, p):
    k = int(np.floor(k))
    if k < 0:
        return 0.0
    return float(np.mean(nbinom.cdf(k, r, p)))


def nbinom_mixture_quantile(prob, r, p):
    lo = -1
    hi = int(np.max(nbinom.ppf(prob, r, p)))
    if not np.isfinite(hi) or hi < 0:
        hi = 1
    while nbinom_mixture_cdf(hi, r, p) < prob:
        hi = 2 * hi + 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if nbinom_mixture_cdf(mid, r, p) >= prob:
            hi = mid
        else:
            lo = mid
    return int(hi)


def interval_score(alpha, lower, upper, y):
    score = upper - lower
    if y < lower:
        score += (2.0 / alpha) * (lower - y)
    elif y > upper:
        score += (2.0 / alpha) * (y - upper)
    return float(score)


def wis(y, mu, vmr, alphas=WIS_ALPHAS):
    r, p = mean_vmr_to_nbinom_params(mu, vmr)
    probs = np.unique(np.concatenate(([0.5], alphas / 2.0, 1.0 - alphas / 2.0)))
    quantiles = {q: nbinom_mixture_quantile(q, r, p) for q in probs}
    median = quantiles[0.5]
    lower = np.array([quantiles[alpha / 2.0] for alpha in alphas], dtype=float)
    upper = np.array([quantiles[1.0 - alpha / 2.0] for alpha in alphas], dtype=float)
    total = 0.5 * abs(float(y) - median)
    for alpha, lo, hi in zip(alphas, lower, upper):
        total += 0.5 * alpha * interval_score(alpha, lo, hi, y)
    return float(total / (len(alphas) + 0.5)), median, lower, upper


def wis_per_timepoint(y_obs, mean_pred, vmr, alphas=WIS_ALPHAS):
    scores = np.empty(len(y_obs), dtype=float)
    medians = np.empty(len(y_obs), dtype=float)
    lower = np.empty((len(alphas), len(y_obs)), dtype=float)
    upper = np.empty((len(alphas), len(y_obs)), dtype=float)
    for t, y in enumerate(y_obs):
        scores[t], medians[t], lower[:, t], upper[:, t] = wis(y, mean_pred[:, t], vmr, alphas)
    return scores, medians, lower, upper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble_predictions_path", required=True)
    parser.add_argument("--phase_cut_date", required=True)
    parser.add_argument("--substance_normalization", required=True)
    parser.add_argument("--gene_target", required=True)
    parser.add_argument("--town", required=True)
    args = parser.parse_args()

    base_config = {
        "data_kwargs": {
            "town": args.town,
            "sampling_area": "North_South",
            "project": "both",
            "max_precipitation_subsetting": None,
            "substance_normalization": args.substance_normalization,
            "gene_target": args.gene_target,
            "log_scale": True,
        },
        "phase_cut_date": args.phase_cut_date,
        "dt": 0.2,
        "T_max": 25,
    }

    data = optimization_utils.two_phase_integrative_model_load_data(base_config)
    ensemble_dir = Path(args.ensemble_predictions_path)
    mean_pred = np.load(ensemble_dir / "ensemble_predictions_I7_reported.npz", allow_pickle=True)["all"]
    vmr = np.load(ensemble_dir / "ensemble_predictions_noise_parameters.npz", allow_pickle=True)["all"][:, 0]

    test_idx = np.asarray(data["t_mask_I_test"])
    y_obs = np.asarray(data["I_test"], dtype=float)
    test_dates = np.asarray(data["obs_dates_phase_2"], dtype="datetime64[D]")
    scores, medians, lower, upper = wis_per_timepoint(y_obs, mean_pred[:, test_idx], vmr, WIS_ALPHAS)

    np.savez(
        ensemble_dir / "weighted_interval_score_I7_reported_test.npz",
        dates=test_dates,
        observations=y_obs,
        alphas=WIS_ALPHAS,
        median=medians,
        interval_lower=lower,
        interval_upper=upper,
        wis=scores,
        mean_wis=np.array(scores.mean()),
    )
    (ensemble_dir / "weighted_interval_score_I7_reported_test_mean.txt").write_text(f"{scores.mean()}\n")
    print(scores.mean())


if __name__ == "__main__":
    main()
