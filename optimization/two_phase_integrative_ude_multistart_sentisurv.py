import jax.random as jr
import jax.numpy as jnp
import jax
from jax.scipy.special import gammaln
import sys
import os, pathlib, tempfile
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import optimization_utils

import optuna
import gc
from datetime import datetime  # added for timestamping saves
import argparse


import data.data_utils_integrative_model as data_utils_integrative_model

import equinox as eqx, json, pathlib
import pandas as pd


parser = argparse.ArgumentParser(description="Run two-phase integrative UDE Optuna optimization.")
parser.add_argument("--phase_cut_date", type=str, required=True, help="Date to cut phases, format YYYY-MM-DD")
parser.add_argument("--town", type=str, required=True, help="Town name")
parser.add_argument("--prev_phase_cut_date", type=str, required=True, help="Date to cut phases, format YYYY-MM-DD")
parser.add_argument("--objective", type=str, required=True, help="Objective function to optimize", default="cases_and_conc")
parser.add_argument("--seed_batch", type=int, required=True, help="Random seed for initialization")
args = parser.parse_args()

base_config = {
        # General settings
        "seed": 0,  # random seed

        # data selection settings
        "data_kwargs": {
            "town": args.town,
            "log_scale": True, # this only considers WW measurements, not case counts
        },
        
        "dt": 0.2,
        "T_max": 25, # dummy value
        "underreporting_model": "monotone_increasing"
}
base_config["phase_cut_date"] = args.phase_cut_date
base_config["prev_phase_cut_date"] = args.prev_phase_cut_date
data = optimization_utils.two_phase_integrative_model_load_data(base_config)

base_config["E0"] = float(data["I_all"][0]/7)*1/0.5
base_config["I0"] = float(data["I_all"][0]/7)*5
base_config["R0"] = 0.92*data["population_size"] # 92% of pop, based on https://www.rki.de/DE/Themen/Infektionskrankheiten/Infektionskrankheiten-A-Z/C/COVID-19-Pandemie/AK-Studien/Ergebnisse.html


def build_config_from_saved(base_config, hparams: dict) -> dict:
    """Merge base config from integrative_ude_optuna.py with saved trial params."""
    import copy

    base = copy.deepcopy(base_config)
    base.update(hparams)
    # Keep solver kwargs consistent
    if "solver_kwargs" not in base:
        base["solver_kwargs"] = {
            "rtol": hparams.get("rtol", 1e-4),
            "atol": hparams.get("atol", 1e-6),
        }
    else:
        base["solver_kwargs"]["rtol"] = hparams.get(
            "rtol", base["solver_kwargs"].get("rtol", 1e-4)
        )
        base["solver_kwargs"]["atol"] = hparams.get(
            "atol", base["solver_kwargs"].get("atol", 1e-6)
        )
    base["lr_strategy"] = (base["lr_step1"],base["lr_step2"], base["lr_step3"],base["lr_step4"]) 
    base["steps_strategy"] = (base["step1"], base["step2"], base["step3"], base["step4"])   
    return base

with open(f"{base_config['data_kwargs']['town']}/optuna_best_{base_config['phase_cut_date']}_prev{base_config['prev_phase_cut_date']}_{args.objective}/hparams.json") as f:
    hparams = json.load(f)
    #print(hparams)

# Merge with base config from your Optuna script
config = build_config_from_saved(base_config, hparams)

def per_observable_likelihood(model, t_all, t_phase_1, t_mask_ids_I, t_mask_ids_conc, t_mask_ids_prev, obs_cases, obs_conc, obs_pos_tests, obs_total_tests):
    pred_conc, I_new_7d_pred, prevalence = model.run_model_with_prevalence_output(t_all, t_phase_1)

    # Select observed timestamps from predictions
    I_new_7d_sel = I_new_7d_pred[t_mask_ids_I]
    conc_sel     = pred_conc[t_mask_ids_conc]
    prev_sel     = prevalence[t_mask_ids_prev]

    # Masks to ignore NaNs from 7-day window/reporting delay and missing obs
    I_valid = (~jnp.isnan(I_new_7d_sel))
    C_valid = (~jnp.isnan(conc_sel))
    P_valid = (~jnp.isnan(prev_sel))

    pred_I_masked = jnp.where(I_valid, I_new_7d_sel, 0.0)
    obs_I_masked  = jnp.where(I_valid, obs_cases, 0.0)
    pred_C_masked = jnp.where(C_valid, conc_sel, 0.0)
    obs_C_masked  = jnp.where(C_valid, obs_conc, 0.0)
    pred_P_masked = jnp.where(P_valid, prev_sel, 0.0)

    # Pull σ from the model
    # sigma_I = jnp.exp(model.log_sigma_I)
    sigma_C = jnp.exp(model.log_sigma_C)
    vmr_I = 1.0 + jax.nn.softplus(model.par_vmr)

    I_mask = jnp.sum(jnp.where(I_valid, 1, 0))
    C_mask = jnp.sum(jnp.where(C_valid, 1, 0))
    P_mask = jnp.sum(jnp.where(P_valid, 1, 0))

    # ----- Negative binomial NLL for case counts -----
    # Convert (mean, VMR) -> (r, p)
    eps = 1e-8
    # p = 1 / VMR  (independent of μ)
    p_nb = jnp.clip(1.0 / vmr_I, eps, 1.0 - eps)
    # r = μ / (VMR - 1)
    r_nb = jnp.clip(pred_I_masked / jnp.maximum(vmr_I - 1.0, eps), eps, 1e12)

    # Count data (assumed integer non-negative)
    k = jnp.clip(obs_I_masked, 0.0, 1e12)

    # log PMF: log C(k+r-1, k) + r log p + k log(1-p)
    logpmf_nb = (
        gammaln(k + r_nb) - gammaln(r_nb) - gammaln(k + 1.0)
        + r_nb * jnp.log(p_nb) + k * jnp.log1p(-p_nb)
    )

    nll_I = -jnp.sum(jnp.where(I_valid, logpmf_nb, 0.0))
    nll_C = 0.5 * jnp.sum(((pred_C_masked - obs_C_masked) / sigma_C) ** 2) + 0.5 * jnp.log(2 * jnp.pi * sigma_C ** 2) * C_mask # this is equivalent to log normal noise model (assuming that the data doesn't change, i.e. up to a constant value) 

    # Binomial negative log-likelihood for prevalence data
    se = 0.83  # sensitivity (used European values of Table 4 in https://pmc.ncbi.nlm.nih.gov/articles/PMC11527648/)
    sp = 1.0 # specificity
    p_obs = se * pred_P_masked + (1.0 - sp) * (1.0 - pred_P_masked)
    nll_prev = -jnp.sum(
        jnp.where(
        P_valid,
        obs_pos_tests * jnp.log(jnp.clip(p_obs, 1e-8, 1.0)) +
        (obs_total_tests - obs_pos_tests) * jnp.log(jnp.clip(1.0 - p_obs, 1e-8, 1.0)),
        0.0
        )
    ) # up to a constant (binomial coefficient)
    # Convert to mean negative log-likelihood per valid entry
    nll_prev = nll_prev / jnp.maximum(P_mask, 1)

    return (nll_I / I_mask), (nll_C / C_mask), nll_prev, (nll_I / I_mask) + (nll_C / C_mask) + nll_prev, I_mask, C_mask, P_mask


data = optimization_utils.two_phase_integrative_model_load_data(config)

def objective(seed):
    hp_config = {
        "seed": seed,
        "k1_init": float(jnp.exp(jr.uniform(jr.PRNGKey(seed), minval=jnp.log(1e-2), maxval=jnp.log(40.0)))),
        "k2_init": float(jr.uniform(jr.PRNGKey(seed+1), minval=0.6, maxval=2.5)),
        "k3_init": float(jr.uniform(jr.PRNGKey(seed+2), minval=0.15, maxval=2.0)),
        "T_peak_init": float(jr.uniform(jr.PRNGKey(seed+3), minval=0.0, maxval=5.0)),  # peak time in days
        "E0": base_config["E0"] + jr.normal(jr.PRNGKey(seed+4), ()) * base_config["E0"] * 0.1,
        "I0": base_config["I0"] + jr.normal(jr.PRNGKey(seed+5), ()) * base_config["I0"] * 0.1,
        "R0": base_config["R0"] + jr.normal(jr.PRNGKey(seed+6), ()) * base_config["R0"] * 0.1,
        "reporting_delay": int(jnp.maximum(0, config["reporting_delay"] + int(jr.randint(jr.PRNGKey(seed+7), (), minval=-2, maxval=2)))),
    }
    config.update(hp_config)

    # Model initialization
    # Train and validate model...
    if args.objective == "cases_and_conc":
        model, total_negll, train_negll, val_negll, train_loss = optimization_utils.two_phase_integrative_model_train_and_evaluate(config, None, "todo", print_every=100)
    elif args.objective == "prev_and_conc":
        model, total_negll, train_negll, val_negll, train_loss = optimization_utils.two_phase_integrative_model_train_and_evaluate_prevalence_objective(config, None, "todo", print_every=100)
    elif args.objective == "three_objectives":
        model, total_negll, train_negll, val_negll, train_loss = optimization_utils.two_phase_integrative_model_train_and_evaluate_three_objectives(config, None, "todo", print_every=100)
    else:
        raise ValueError(f"Unknown objective {args.objective}")

    train_negll_I, train_negll_c, train_negll_prev, train_negll, n_obs_train_I, n_obs_train_c, n_obs_train_prev = per_observable_likelihood(model, data["t_all"], data["t_phase_1"], data["t_mask_I_train"], data["t_mask_conc_train"], data["t_mask_prev_train"],data["I_train"], data["conc_train"], data["pos_tests_train"], data["n_tests_train"])
    val_negll_I, val_negll_c, val_negll_prev, val_negll, n_obs_val_I, n_obs_val_c, n_obs_val_prev = per_observable_likelihood(model, data["t_all"], data["t_phase_1"], data["t_mask_I_val"], data["t_mask_conc_val"], data["t_mask_prev_val"], data["I_val"], data["conc_val"], data["pos_tests_val"], data["n_tests_val"])
    total_negll_I, total_negll_c, total_negll_prev, total_negll, n_obs_total_I, n_obs_total_c, n_obs_total_prev = per_observable_likelihood(model, data["t_all"], data["t_phase_1"], data["t_mask_I_all"], data["t_mask_conc_all"], data["t_mask_prev_all"], data["I_all"], data["eval_conc"], data["pos_tests"], data["n_tests"])
    test_negll_I, _, test_negll_prev, _, n_obs_test_I, _, n_obs_test_prev = per_observable_likelihood(model, data["t_all"], data["t_phase_1"], data["t_mask_I_test"], data["t_mask_conc_all"], data["t_mask_prev_test"], data["I_test"], data["eval_conc"], data["pos_tests_test"], data["n_tests_test"])


    outdir = pathlib.Path(f"{config['data_kwargs']['town']}/multistart_models/{config['phase_cut_date']}_prev{config['prev_phase_cut_date']}_{args.objective}")
    outdir.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(outdir / f"{seed}_model.eqx", model)
    meta = {
                "train_negll": float(train_negll),
                "val_negll": float(val_negll),
                "total_negll": float(total_negll),
                "train_negll_I": float(train_negll_I),
                "val_negll_I": float(val_negll_I),
                "test_negll_I": float(test_negll_I),
                "total_negll_I": float(total_negll_I),
                "train_negll_c": float(train_negll_c),
                "val_negll_c": float(val_negll_c),
                "total_negll_c": float(total_negll_c),
                "train_negll_prev": float(train_negll_prev),
                "val_negll_prev": float(val_negll_prev),
                "test_negll_prev": float(test_negll_prev),
                "total_negll_prev": float(total_negll_prev),
                "n_obs_train_I": int(n_obs_train_I),
                "n_obs_train_c": int(n_obs_train_c),
                "n_obs_train_prev": int(n_obs_train_prev),
                "n_obs_val_I": int(n_obs_val_I),
                "n_obs_val_c": int(n_obs_val_c),
                "n_obs_val_prev": int(n_obs_val_prev),
                "n_obs_total_I": int(n_obs_total_I),
                "n_obs_total_c": int(n_obs_total_c),
                "n_obs_total_prev": int(n_obs_total_prev),
                "n_obs_test_I": int(n_obs_test_I),
                "n_obs_test_prev": int(n_obs_test_prev),
                "k1": float(jnp.exp(model.log_k1)),
                "k2": float(jax.nn.sigmoid(model.logit_k2)*(2.5-0.6) + 0.6),
                "k3": float(jax.nn.sigmoid(model.logit_k3)*(2.0-0.15) + 0.15),
                "E0": model.UDE.E0.item(),
                "I0": model.UDE.I0.item(),
                "R0": model.UDE.R0.item(),
                "T_peak": float(jax.nn.sigmoid(model.logit_T_peak)*(5-1)+1),
                "reporting_delay": int(model.reporting_delay),
            }
    (outdir / f"{seed}_metrics.json").write_text(json.dumps(meta, indent=2))

    gc.collect()    


seed_batch = args.seed_batch

# for every seed batch, evaluate 30 seeds
for seed in range(seed_batch*30, (seed_batch+1)*30):
    print(f"Starting optimization with seed {seed}", flush=True)
    objective(seed)