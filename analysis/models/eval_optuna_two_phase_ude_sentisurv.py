import jax.random as jr
import jax.numpy as jnp
import jax
import sys
import os
import json

from optimization import optimization_utils
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))

import model_definition.two_phase_integrative_ude as two_phase_integrative_ude
import data.data_utils_integrative_model as data_utils_integrative_model
import model_definition.model_utils as model_utils
from jax.scipy.special import gammaln

import equinox as eqx
import pandas as pd
import argparse
import optuna
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from optuna.visualization import (
    plot_param_importances, 
)
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="Run two-phase integrative UDE Optuna optimization.")
parser.add_argument("--phase_cut_date", type=str, required=True, help="Date to cut phases, format YYYY-MM-DD")
parser.add_argument("--prev_phase_cut_date", type=str, required=True, help="Date to cut phases, format YYYY-MM-DD")
parser.add_argument("--town", type=str, required=True, help="Town name")
parser.add_argument("--objective", type=str, required=True, help="Objective function to optimize", default="cases_and_conc")
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
        "underreporting_model": "monotone_increasing",
}
base_config["phase_cut_date"] = args.phase_cut_date
base_config["prev_phase_cut_date"] = args.prev_phase_cut_date
data = optimization_utils.two_phase_integrative_model_load_data(base_config)

base_config["E0"] = float(data["I_all"][0]/7)*1/0.5
base_config["I0"] = float(data["I_all"][0]/7)*5
base_config["R0"] = 0.92*data["population_size"] # 92% of pop, based on https://www.rki.de/DE/Themen/Infektionskrankheiten/Infektionskrankheiten-A-Z/C/COVID-19-Pandemie/AK-Studien/Ergebnisse.html


model_path = pathlib.Path(f"{base_config['data_kwargs']['town']}/optuna_best_{base_config['phase_cut_date']}_prev{base_config['prev_phase_cut_date']}_{args.objective}")

def build_config_from_saved(base_config, hparams: dict) -> dict:
    """Merge base config from integrative_ude_optuna.py with saved trial params."""
    import copy

    base = copy.deepcopy(base_config)
    base.update(hparams)
    # Keep solver kwargs consistent
    if "solver_kwargs" not in base:
        base["solver_kwargs"] = {
            "rtol": hparams.get("rtol"),
            "atol": hparams.get("atol"),
        }
    else:
        base["solver_kwargs"]["rtol"] = hparams.get(
            "rtol", base["solver_kwargs"].get("rtol")
        )
        base["solver_kwargs"]["atol"] = hparams.get(
            "atol", base["solver_kwargs"].get("atol")
        )
    return base

with open(f"{model_path}/hparams.json") as f:
    hparams = json.load(f)

# Merge with base config from your Optuna script
config = build_config_from_saved(base_config, hparams)

# Generate dataset
data = optimization_utils.two_phase_integrative_model_load_data(config)

model = two_phase_integrative_ude.IntegrativeModel(
    width_size=config["width_size"],
    depth=config["depth"],
    activation=model_utils.activation_fct_mapper[config["activation"]],
    underreporting_model=config["underreporting_model"],
    n_freqs=config["n_freqs"],
    t_scale=data["t_scale"],
    population_size=data["population_size"],
    E0_init=config["E0"],
    I0_init=config["I0"],
    R0_init=config["R0"],
    solver=model_utils.solver_mapper[config["solver"]],
    solver_kwargs=config["solver_kwargs"],
    k1_init=config["k1_init"],
    k2_init=config["k2_init"],
    k3_init=config["k3_init"],
    T_peak_init=config["T_peak_init"],
    T_max=config["T_max"],
    dt = config["dt"],
    key=jr.key(config["seed"]),
    init_par_vmr=config.get("init_par_vmr", 1.0),
    init_sigma_C=config.get("init_sigma_C", 1.0),
    reporting_delay=config.get("reporting_delay", 3)) # days, delay between infection and reporting

model = eqx.tree_deserialise_leaves(f"{model_path}/model.eqx", model)


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

    return (nll_I / I_mask), (nll_C / C_mask), nll_prev, (nll_I / I_mask) + (nll_C / C_mask) + nll_prev

train_negll_I, train_negll_c, train_negll_prev, train_negll = per_observable_likelihood(model, data["t_all"], data["t_phase_1"], data["t_mask_I_train"], data["t_mask_conc_train"], data["t_mask_prev_train"],data["I_train"], data["conc_train"], data["pos_tests_train"], data["n_tests_train"])
val_negll_I, val_negll_c, val_negll_prev, val_negll = per_observable_likelihood(model, data["t_all"], data["t_phase_1"], data["t_mask_I_val"], data["t_mask_conc_val"], data["t_mask_prev_val"], data["I_val"], data["conc_val"], data["pos_tests_val"], data["n_tests_val"])
total_negll_I, total_negll_c, total_negll_prev, total_negll = per_observable_likelihood(model, data["t_all"], data["t_phase_1"], data["t_mask_I_all"], data["t_mask_conc_all"], data["t_mask_prev_all"], data["I_all"], data["eval_conc"], data["pos_tests"], data["n_tests"])
test_negll_I, _, test_negll_prev, _ = per_observable_likelihood(model, data["t_all"], data["t_phase_1"], data["t_mask_I_test"], data["t_mask_conc_all"], data["t_mask_prev_test"], data["I_test"], data["eval_conc"], data["pos_tests_test"], data["n_tests_test"])
print(f"Model 1: train_negll: {train_negll}, val_negll: {val_negll}, total_negll: {total_negll}", flush=True)
print(f"Model 1: train_negll_I: {train_negll_I}, val_negll_I: {val_negll_I}, total_negll_I: {total_negll_I}", flush=True)
print(f"Model 1: train_negll_c: {train_negll_c}, val_negll_c: {val_negll_c}, total_negll_c: {total_negll_c}", flush=True)

meta = {
            "train_negll": float(train_negll),
            "val_negll": float(val_negll),
            "total_negll": float(total_negll),
            "train_negll_I": float(train_negll_I),
            "val_negll_I": float(val_negll_I),
            "test_negll_I": float(test_negll_I),
            "total_negll_I": float(total_negll_I),
            "train_negll_prev": float(train_negll_prev),
            "val_negll_prev": float(val_negll_prev),
            "test_negll_prev": float(test_negll_prev),
            "total_negll_prev": float(total_negll_prev),
            "train_negll_c": float(train_negll_c),
            "val_negll_c": float(val_negll_c),
            "total_negll_c": float(total_negll_c),
            "k1": float(jnp.exp(model.log_k1)),
            "k2": float(jax.nn.sigmoid(model.logit_k2)*(2.5-0.6) + 0.6),
            "k3": float(jax.nn.sigmoid(model.logit_k3)*(2.0-0.15) + 0.15),
            "T_peak": float(jax.nn.sigmoid(model.logit_T_peak)*(5-1)+1),
            "reporting_delay": int(model.reporting_delay),
        }
(model_path / f"1_metrics.json").write_text(json.dumps(meta, indent=2))

fig1, fig2 = two_phase_integrative_ude.plot_model(model, data["t_all"], data["t_phase_1"], config["phase_cut_date"],
               data["I_dates_train"], data["I_train"],
               data["I_dates_val"], data["I_val"],
               data["obs_dates_phase_2"], data["I_test"], data["dates_all"],
               data["conc_dates_train"], data["conc_train"],
               data["conc_dates_val"], data["conc_val"], data["t_all_idx"], conc_y_axis_label="Log PMMoV normalized\nconcentration")
fig1.savefig(f"{model_path}/1_I_model_fit.png", dpi=300)
fig2.savefig(f"{model_path}/1_c_model_fit.png", dpi=300)

fig3 = two_phase_integrative_ude.plot_SEIR_prediction(model, data["t_all"], data["dates_all"], config["phase_cut_date"])
fig3.savefig(f"{model_path}/1_SEIR_prediction.png", dpi=300)

fig4 = two_phase_integrative_ude.plot_beta_prediction(model, data["t_all"], data["dates_all"], config["phase_cut_date"])
fig4.savefig(f"{model_path}/1_beta_prediction.png", dpi=300)

fig5 = two_phase_integrative_ude.plot_effective_reproduction_number(model, data["t_all"], data["dates_all"], config["phase_cut_date"])
fig5.savefig(f"{model_path}/1_effective_reproduction_number.png", dpi=300)

fig6 = two_phase_integrative_ude.plot_underreporting(model, data["t_all"], data["t_phase_1"], data["dates_all"], config["phase_cut_date"])
fig6.savefig(f"{model_path}/1_underreporting.png", dpi=300)

fig7 = two_phase_integrative_ude.plot_shedding_curve(model)
fig7.savefig(f"{model_path}/1_shedding_curve.png", dpi=300)

fig8 = two_phase_integrative_ude.plot_test_positive_rate(model, data["t_all"], data["dates_all"], config["prev_phase_cut_date"], config["data_kwargs"]["town"], data)
fig8.savefig(f"{model_path}/1_test_positive_rate.png", dpi=300)

# now evaluate the 10 best models
import copy
def get_hp_config_from_trial(study_df_row, base_config):
    base = copy.deepcopy(base_config)
    base.update(hparams)
    hp_config = {
        "trial_number": int(study_df_row["number"]),
        # NN settings
        "width_size": int(study_df_row["params_width_size"]), # width of NN
        "depth": int(study_df_row["params_depth"]), # depth of NN
        "activation": study_df_row["params_activation"], # activation function
        "n_freqs": int(study_df_row["params_n_freqs"]), # number of frequencies for additional Fourier features

        "T_peak_init": float(study_df_row["params_T_peak_init"]),  # peak time in days
        "T_max": int(study_df_row["params_T_max"]),  # max time in days
        "k1_init": float(study_df_row["params_k1_init"]),  # initial shedding curve parameter
        "k2_init": float(study_df_row["params_k2_init"]),
        "k3_init": float(study_df_row["params_k3_init"]),

        "init_par_vmr": float(study_df_row["params_init_par_vmr"]),  # initial value for the log‐noise of cases
        "init_sigma_C": float(study_df_row["params_init_sigma_C"]),  # initial value for the log‐noise of concentration

        "reporting_delay": int(study_df_row["params_reporting_delay"]),  # days, delay between infection and reporting

        # solver settings
        "solver": study_df_row["params_solver"],  # solver to use
        "solver_kwargs": {
            "rtol": float(study_df_row["params_rtol"]),
            "atol": float(study_df_row["params_atol"]),
        },

        # Optimization settings
        "lr_strategy": (float(study_df_row["params_lr_step1"]), 
                            float(study_df_row["params_lr_step2"]), 
                            float(study_df_row["params_lr_step3"]),
                            float(study_df_row["params_lr_step4"])), # lr for each strategy (1e-4, 5e-5, 3e-5, 1e-5),
        "steps_strategy": (int(study_df_row["params_step1"]), 
                            int(study_df_row["params_step2"]), 
                            int(study_df_row["params_step3"]), 
                            int(study_df_row["params_step4"])), # trial.suggest_int("step4_lbfgs", 0, 2000), ), # epochs for each strategy
        "reg_norm": float(study_df_row["params_reg_norm"]), # weight decay parameter
        "regularization_mode": study_df_row["params_regularization_mode"], # one of L2, beta_derivative
    }
    hp_config["par_T_peak"] = float(jnp.log((hp_config["T_peak_init"]/hp_config["T_max"]) / (1 - (hp_config["T_peak_init"]/hp_config["T_max"])))) 
    base.update(hp_config)
    return base

storage = f"sqlite:///{config['data_kwargs']['town']}/optuna_study_two_phase_model_{config['phase_cut_date']}_prev{config['prev_phase_cut_date']}_{args.objective}.db"

study = optuna.create_study(
    study_name="ude_hp_search_cc_ude",
    direction="minimize",
    storage=storage,  # Save to local SQLite DB
    load_if_exists=True)

df_optuna = study.trials_dataframe()
best_10_models = df_optuna.sort_values("user_attrs_val_nll").iloc[:10]

for i in range(1, 10):
    os.makedirs(model_path / f"best_10", exist_ok=True)
    new_config = get_hp_config_from_trial(best_10_models.iloc[i], base_config)
    
    if args.objective == "cases_and_conc":
        model, total_negll, train_negll, val_negll, train_loss = optimization_utils.two_phase_integrative_model_train_and_evaluate(new_config, None, "todo", print_every=100, trial=new_config["trial_number"])
    elif args.objective == "prev_and_conc":
        model, total_negll, train_negll, val_negll, train_loss = optimization_utils.two_phase_integrative_model_train_and_evaluate_prevalence_objective(new_config, None, "todo", print_every=100, trial=new_config["trial_number"])
    elif args.objective == "three_objectives":
        model, total_negll, train_negll, val_negll, train_loss = optimization_utils.two_phase_integrative_model_train_and_evaluate_three_objectives(new_config, None, "todo", print_every=100, trial=new_config["trial_number"])
    else:
        raise ValueError(f"Unknown objective {args.objective}")
    
    train_negll_I, train_negll_c, train_negll_prev, train_negll = per_observable_likelihood(model, data["t_all"], data["t_phase_1"], data["t_mask_I_train"], data["t_mask_conc_train"], data["t_mask_prev_train"],data["I_train"], data["conc_train"], data["pos_tests_train"], data["n_tests_train"])
    val_negll_I, val_negll_c, val_negll_prev, val_negll = per_observable_likelihood(model, data["t_all"], data["t_phase_1"], data["t_mask_I_val"], data["t_mask_conc_val"], data["t_mask_prev_val"], data["I_val"], data["conc_val"], data["pos_tests_val"], data["n_tests_val"])
    total_negll_I, total_negll_c, total_negll_prev, total_negll = per_observable_likelihood(model, data["t_all"], data["t_phase_1"], data["t_mask_I_all"], data["t_mask_conc_all"], data["t_mask_prev_all"], data["I_all"], data["eval_conc"], data["pos_tests"], data["n_tests"])
    test_negll_I, _, test_negll_prev, _ = per_observable_likelihood(model, data["t_all"], data["t_phase_1"], data["t_mask_I_test"], data["t_mask_conc_all"], data["t_mask_prev_test"], data["I_test"], data["eval_conc"], data["pos_tests_test"], data["n_tests_test"])
    print(f"Model {i}: train_negll: {train_negll}, val_negll: {val_negll}, total_negll: {total_negll}", flush=True)
    print(f"Model {i}: train_negll_I: {train_negll_I}, val_negll_I: {val_negll_I}, total_negll_I: {total_negll_I}", flush=True)
    print(f"Model {i}: train_negll_c: {train_negll_c}, val_negll_c: {val_negll_c}, total_negll_c: {total_negll_c}", flush=True)

    fig1, fig2 = two_phase_integrative_ude.plot_model(model, data["t_all"], data["t_phase_1"], new_config["phase_cut_date"],
               data["I_dates_train"], data["I_train"],
               data["I_dates_val"], data["I_val"],
               data["obs_dates_phase_2"], data["I_test"], data["dates_all"],
               data["conc_dates_train"], data["conc_train"],
               data["conc_dates_val"], data["conc_val"], data["t_all_idx"], conc_y_axis_label="Log PMMoV normalized\nconcentration")
    fig1.savefig(f"{model_path}/best_10/{i+1}_I_model_fit.png", dpi=300)
    fig2.savefig(f"{model_path}/best_10/{i+1}_c_model_fit.png", dpi=300)

    fig3 = two_phase_integrative_ude.plot_SEIR_prediction(model, data["t_all"], data["dates_all"], new_config["phase_cut_date"])
    fig3.savefig(f"{model_path}/best_10/{i+1}_SEIR_prediction.png", dpi=300)

    fig4 = two_phase_integrative_ude.plot_beta_prediction(model, data["t_all"], data["dates_all"], new_config["phase_cut_date"])
    fig4.savefig(f"{model_path}/best_10/{i+1}_beta_prediction.png", dpi=300)

    fig5 = two_phase_integrative_ude.plot_effective_reproduction_number(model, data["t_all"], data["dates_all"], new_config["phase_cut_date"])
    fig5.savefig(f"{model_path}/best_10/{i+1}_effective_reproduction_number.png", dpi=300)

    fig6 = two_phase_integrative_ude.plot_underreporting(model, data["t_all"], data["t_phase_1"], data["dates_all"], new_config["phase_cut_date"])
    fig6.savefig(f"{model_path}/best_10/{i+1}_underreporting.png", dpi=300)

    fig7 = two_phase_integrative_ude.plot_shedding_curve(model)
    fig7.savefig(f"{model_path}/best_10/{i+1}_shedding_curve.png", dpi=300)

    fig8 = two_phase_integrative_ude.plot_test_positive_rate(model, data["t_all"], data["dates_all"], config["prev_phase_cut_date"], config["data_kwargs"]["town"], data)
    fig8.savefig(f"{model_path}/best_10/{i+1}_test_positive_rate.png", dpi=300)
    
    eqx.tree_serialise_leaves(model_path / f"best_10/{i+1}_model.eqx", model)
    meta = {
                "trial_number": int(new_config["trial_number"]),
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
                "train_loss": float(train_loss), 
                "k1": float(jnp.exp(model.log_k1)),
                "k2": float(jax.nn.sigmoid(model.logit_k2)*(2.5-0.6) + 0.6),
                "k3": float(jax.nn.sigmoid(model.logit_k3)*(2.0-0.15) + 0.15),
                "T_peak": float(jax.nn.sigmoid(model.logit_T_peak)*(5-1)+1),
                "reporting_delay": int(model.reporting_delay),
            }
    (model_path / f"best_10/{i+1}_metrics.json").write_text(json.dumps(meta, indent=2))
