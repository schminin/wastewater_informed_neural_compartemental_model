import argparse
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binom

sys.path.append(str(Path(__file__).resolve().parents[2]))
from optimization import optimization_utils


WIS_ALPHAS = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=float)


def observed_positive_probability(pred_p, se=0.83, sp=1.0, eps=1e-8):
    pred_p = np.asarray(pred_p, dtype=float)

    if np.any((pred_p < 0.0) | (pred_p > 1.0)):
        raise ValueError("All predicted probabilities must lie in [0, 1].")
    if not (0.0 <= se <= 1.0):
        raise ValueError("se must lie in [0, 1].")
    if not (0.0 <= sp <= 1.0):
        raise ValueError("sp must lie in [0, 1].")

    p_obs = se * pred_p + (1.0 - sp) * (1.0 - pred_p)
    return np.clip(p_obs, eps, 1.0 - eps)


def binomial_mixture_cdf(k, n, p):
    k = int(np.floor(k))
    n = int(n)

    if n < 0:
        raise ValueError("n must be non-negative.")
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0

    p = np.asarray(p, dtype=float)
    if p.ndim != 1:
        raise ValueError("p must be a 1D array.")
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("All entries of p must satisfy 0 < p < 1.")

    return float(np.mean(binom.cdf(k, n, p)))


def binomial_mixture_quantile(prob, n, p):
    if not (0.0 < prob < 1.0):
        raise ValueError("prob must satisfy 0 < prob < 1.")

    n = int(n)
    if n < 0:
        raise ValueError("n must be non-negative.")

    p = np.asarray(p, dtype=float)
    if p.ndim != 1:
        raise ValueError("p must be a 1D array.")
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("All entries of p must satisfy 0 < p < 1.")

    lo = -1
    hi = n

    while hi - lo > 1:
        mid = (lo + hi) // 2
        if binomial_mixture_cdf(mid, n, p) >= prob:
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


def wis_from_binomial_probs(y, n, p, alphas=WIS_ALPHAS):
    y = float(y)
    n = int(n)
    p = np.asarray(p, dtype=float)
    alphas = np.asarray(alphas, dtype=float)

    if y < 0.0 or y > n:
        raise ValueError("Observed count y must satisfy 0 <= y <= n.")
    if np.any((alphas <= 0.0) | (alphas >= 1.0)):
        raise ValueError("All alphas must satisfy 0 < alpha < 1.")

    probs = np.unique(np.concatenate(([0.5], alphas / 2.0, 1.0 - alphas / 2.0)))
    quantiles = {q: binomial_mixture_quantile(q, n, p) for q in probs}

    median = quantiles[0.5]
    lower = np.array([quantiles[alpha / 2.0] for alpha in alphas], dtype=float)
    upper = np.array([quantiles[1.0 - alpha / 2.0] for alpha in alphas], dtype=float)

    total = 0.5 * abs(y - median)
    for alpha, lo, hi in zip(alphas, lower, upper):
        total += 0.5 * alpha * interval_score(alpha, lo, hi, y)

    return float(total / (len(alphas) + 0.5)), median, lower, upper


def wis_from_observed_probs(y, n, p_obs, alphas=WIS_ALPHAS, eps=1e-8):
    p_obs = np.asarray(p_obs, dtype=float)

    if p_obs.ndim != 1:
        raise ValueError("p_obs must be a 1D array.")
    if np.any((p_obs < 0.0) | (p_obs > 1.0)):
        raise ValueError("All entries of p_obs must lie in [0, 1].")

    return wis_from_binomial_probs(y, n, np.clip(p_obs, eps, 1.0 - eps), alphas=alphas)


def wis(y, n, pred_p, se=0.83, sp=1.0, alphas=WIS_ALPHAS, eps=1e-8):
    p_obs = observed_positive_probability(pred_p, se=se, sp=sp, eps=eps)
    return wis_from_binomial_probs(y, n, p_obs, alphas=alphas)


def _validate_timepoint_inputs(obs_pos_tests, obs_total_tests, probs):
    obs_pos_tests = np.asarray(obs_pos_tests, dtype=float)
    obs_total_tests = np.asarray(obs_total_tests, dtype=float)
    probs = np.asarray(probs, dtype=float)

    if obs_pos_tests.ndim != 1:
        raise ValueError("obs_pos_tests must be a 1D array of shape (T,).")
    if obs_total_tests.ndim != 1:
        raise ValueError("obs_total_tests must be a 1D array of shape (T,).")
    if probs.ndim != 2:
        raise ValueError("Forecast probabilities must be a 2D array of shape (M, T).")
    if probs.shape[1] != obs_pos_tests.shape[0]:
        raise ValueError("The second dimension of the forecast probabilities must match len(obs_pos_tests).")
    if obs_total_tests.shape[0] != obs_pos_tests.shape[0]:
        raise ValueError("obs_total_tests must have the same length as obs_pos_tests.")
    if np.any(obs_total_tests < 0.0):
        raise ValueError("All entries of obs_total_tests must be non-negative.")
    if np.any(obs_pos_tests < 0.0):
        raise ValueError("All entries of obs_pos_tests must be non-negative.")
    if np.any(obs_pos_tests > obs_total_tests):
        raise ValueError("Each positive count must be at most the corresponding total test count.")
    if np.any((probs < 0.0) | (probs > 1.0)):
        raise ValueError("All forecast probabilities must lie in [0, 1].")

    return obs_pos_tests, obs_total_tests, probs


def wis_per_timepoint_from_observed_probs(obs_pos_tests, obs_total_tests, p_obs, alphas=WIS_ALPHAS, eps=1e-8):
    obs_pos_tests, obs_total_tests, p_obs = _validate_timepoint_inputs(obs_pos_tests, obs_total_tests, p_obs)
    p_obs = np.clip(p_obs, eps, 1.0 - eps)

    scores = np.empty(len(obs_pos_tests), dtype=float)
    medians = np.empty(len(obs_pos_tests), dtype=float)
    lower = np.empty((len(alphas), len(obs_pos_tests)), dtype=float)
    upper = np.empty((len(alphas), len(obs_pos_tests)), dtype=float)

    for t, y in enumerate(obs_pos_tests):
        scores[t], medians[t], lower[:, t], upper[:, t] = wis_from_binomial_probs(
            y=y,
            n=int(obs_total_tests[t]),
            p=p_obs[:, t],
            alphas=alphas,
        )

    return scores, medians, lower, upper


def wis_per_timepoint(obs_pos_tests, obs_total_tests, pred_p, se=0.83, sp=1.0, alphas=WIS_ALPHAS, eps=1e-8):
    p_obs = observed_positive_probability(pred_p, se=se, sp=sp, eps=eps)
    return wis_per_timepoint_from_observed_probs(obs_pos_tests, obs_total_tests, p_obs, alphas=alphas, eps=eps)


def mean_wis_from_observed_probs(obs_pos_tests, obs_total_tests, p_obs, alphas=WIS_ALPHAS, eps=1e-8):
    scores, _, _, _ = wis_per_timepoint_from_observed_probs(
        obs_pos_tests=obs_pos_tests,
        obs_total_tests=obs_total_tests,
        p_obs=p_obs,
        alphas=alphas,
        eps=eps,
    )
    return float(np.mean(scores))


def mean_wis(obs_pos_tests, obs_total_tests, pred_p, se=0.83, sp=1.0, alphas=WIS_ALPHAS, eps=1e-8):
    scores, _, _, _ = wis_per_timepoint(
        obs_pos_tests=obs_pos_tests,
        obs_total_tests=obs_total_tests,
        pred_p=pred_p,
        se=se,
        sp=sp,
        alphas=alphas,
        eps=eps,
    )
    return float(np.mean(scores))


def infer_prev_phase_cut_date(ensemble_predictions_path, phase_cut_date):
    match = re.search(r"_prev(\d{4}-\d{2}-\d{2})_", str(ensemble_predictions_path))
    return match.group(1) if match else phase_cut_date


def normalize_saved_test_positive_rate(pred_rate, eps=1e-8):
    pred_rate = np.asarray(pred_rate, dtype=float)

    if pred_rate.ndim != 2:
        raise ValueError("Saved test-positive-rate forecasts must have shape (M, T).")
    if np.any(pred_rate < 0.0):
        raise ValueError("Saved test-positive-rate forecasts must be non-negative.")

    if np.nanmax(pred_rate) > 1.0 + 1e-6:
        pred_rate = pred_rate / 100.0

    if np.any((pred_rate < 0.0) | (pred_rate > 1.0 + 1e-6)):
        raise ValueError("Saved test-positive-rate forecasts could not be mapped to probabilities in [0, 1].")

    return np.clip(pred_rate, eps, 1.0 - eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble_predictions_path", required=True)
    parser.add_argument("--phase_cut_date", required=True)
    parser.add_argument("--substance_normalization", required=True)
    parser.add_argument("--gene_target", required=True)
    parser.add_argument("--town", required=True)
    parser.add_argument("--prev_phase_cut_date", required=False, default=None)
    args = parser.parse_args()

    if args.town == "Bonn":
        raise ValueError("Binomial WIS is only defined for sentisurv experiments with prevalence observations (town != 'Bonn').")

    ensemble_dir = Path(args.ensemble_predictions_path)
    prev_phase_cut_date = args.prev_phase_cut_date or infer_prev_phase_cut_date(ensemble_dir, args.phase_cut_date)

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
        "prev_phase_cut_date": prev_phase_cut_date,
        "dt": 0.2,
        "T_max": 25,
    }

    data = optimization_utils.two_phase_integrative_model_load_data(base_config)
    test_idx = np.asarray(data["t_mask_prev_test"])
    y_obs = np.asarray(data["pos_tests_test"], dtype=float)
    n_obs = np.asarray(data["n_tests_test"], dtype=float)
    test_dates = np.asarray(data["prevalence_dates_test"], dtype="datetime64[D]")

    if test_idx.size == 0 or y_obs.size == 0:
        raise ValueError("No prevalence test observations were found for the requested sentisurv experiment.")

    pred_rate = np.load(ensemble_dir / "ensemble_predictions_test_positive_rate.npz", allow_pickle=True)["all"]
    pred_probs = normalize_saved_test_positive_rate(pred_rate)[:, test_idx]
    scores, medians, lower, upper = wis_per_timepoint_from_observed_probs(y_obs, n_obs, pred_probs, WIS_ALPHAS)

    observed_rates = np.divide(
        y_obs,
        n_obs,
        out=np.full_like(y_obs, np.nan, dtype=float),
        where=n_obs > 0,
    )

    np.savez(
        ensemble_dir / "weighted_interval_score_test_positive_rate_test.npz",
        dates=test_dates,
        observations=y_obs,
        n_tests=n_obs,
        observed_rate=observed_rates,
        alphas=WIS_ALPHAS,
        median=medians,
        interval_lower=lower,
        interval_upper=upper,
        wis=scores,
        mean_wis=np.array(scores.mean()),
    )
    (ensemble_dir / "weighted_interval_score_test_positive_rate_test_mean.txt").write_text(f"{scores.mean()}\n")
    print(scores.mean())


if __name__ == "__main__":
    main()
