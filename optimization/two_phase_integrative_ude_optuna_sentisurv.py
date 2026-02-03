import jax.random as jr
import jax.numpy as jnp
import jax
import sys
import os, pathlib, tempfile
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import optimization_utils

import optuna
import gc
from datetime import datetime  # added for timestamping saves
import argparse


parser = argparse.ArgumentParser(description="Run two-phase integrative UDE Optuna optimization.")
parser.add_argument("--phase_cut_date", type=str, required=True, help="Date to cut phases, format YYYY-MM-DD")
parser.add_argument("--prev_phase_cut_date", type=str, required=True, help="Date to cut phases, format YYYY-MM-DD")
parser.add_argument("--town", type=str, required=True, help="Town name")
parser.add_argument("--objective", type=str, required=True, help="Objective function to optimize", default="cases_and_conc")
args = parser.parse_args()

config = {
        # General settings
        "seed": 0,  # random seed

        # data selection settings
        "data_kwargs": {
            "town": args.town,
            "log_scale": True, # this only considers WW measurements, not case counts
        },
        
        "dt": 0.2,
        "T_max": 25, # dummy value
}
config["phase_cut_date"] = args.phase_cut_date
config["prev_phase_cut_date"] = args.prev_phase_cut_date
data = optimization_utils.two_phase_integrative_model_load_data(config)

config["E0"] = float(data["I_all"][0]/7)*1/0.5
config["I0"] = float(data["I_all"][0]/7)*5
config["R0"] = 0.92*data["population_size"] # 92% of pop, based on https://www.rki.de/DE/Themen/Infektionskrankheiten/Infektionskrankheiten-A-Z/C/COVID-19-Pandemie/AK-Studien/Ergebnisse.html


def objective(trial):
    hp_config = {
        # NN settings
        "width_size": trial.suggest_int("width_size", 4, 32), # width of NN
        "depth": trial.suggest_int("depth", 1, 3), # depth of NN
        "activation": trial.suggest_categorical("activation", ["tanh", "sigmoid", ]), # activation function
        "n_freqs": trial.suggest_int("n_freqs", 0, 3), # number of frequencies for additional Fourier features

        "T_max": trial.suggest_int("T_max", 12, 28),  # max time in days
        "T_peak_init": trial.suggest_float("T_peak_init", 1, 5),  # peak time in days
        "k1_init": trial.suggest_float("k1_init", 0.01, 40.0, log=True),  # initial shedding curve parameter
        "k2_init": trial.suggest_float("k2_init", 0.6, 2.5),  # initial shedding curve parameter
        "k3_init": trial.suggest_float("k3_init", 0.15, 2.0),  # initial shedding curve parameter

        "init_par_vmr": trial.suggest_float("init_par_vmr", 0.01, 10.0, log=True),  # initial value for the VMR of cases
        "init_sigma_C": trial.suggest_float("init_sigma_C", 0.001, 1.0, log=True),  # initial value for the log‐noise of concentration

        "reporting_delay": trial.suggest_int("reporting_delay", 0, 10),  # days, delay between infection and reporting

        # solver settings
        "solver": trial.suggest_categorical("solver", ["Dopri5", "Tsit5"]),  # solver to use
        "solver_kwargs": {
            "rtol": trial.suggest_float("rtol", 1e-4, 1e-2, log=True),
            "atol": trial.suggest_float("atol", 1e-5, 1e-3, log=True),
        },

        # Optimization settings
        "lr_strategy": (trial.suggest_float("lr_step1", 1e-4, 1e-1, log=True), 
                           trial.suggest_float("lr_step2", 1e-5, 1e-2, log=True), 
                           trial.suggest_float("lr_step3", 1e-5, 1e-2, log=True),
                           trial.suggest_float("lr_step4", 1e-5, 1e-2, log=True),), # lr for each strategy (1e-4, 5e-5, 3e-5, 1e-5),
        "steps_strategy": (trial.suggest_int("step1", 500, 3000), 
                           trial.suggest_int("step2", 500, 3000), 
                           trial.suggest_int("step3", 500, 3000), 
                           trial.suggest_int("step4", 500, 4000)), # trial.suggest_int("step4_lbfgs", 0, 2000), ), # epochs for each strategy
        # regularization of beta NN
        "regularization_mode": trial.suggest_categorical("regularization_mode", ["beta_derivative"]), # one of L2, beta_derivative
        "reg_norm": trial.suggest_float("reg_norm", 1e-5, 1e-3, log=True), # L2 regularization strength
        # regularization of underreporting NN
        "underreporting_regularization_mode": trial.suggest_categorical("underreporting_regularization_mode", ["derivative"]), # one of L2, beta_derivative
        "underreporting_reg_norm": trial.suggest_float("underreporting_reg_norm", 1e-5, 1e-3, log=True), # L2 regularization strength
    }
    config.update(hp_config)
    config["underreporting_model"] = "monotone_increasing"
    
    # Model initialization
    # Train and validate model...
    if args.objective == "cases_and_conc":
        model, total_nll, train_nll, val_nll, train_loss = optimization_utils.two_phase_integrative_model_train_and_evaluate(config, None, "todo", print_every=100, trial=trial.number)    
    elif args.objective == "prev_and_conc":
        model, total_nll, train_nll, val_nll, train_loss = optimization_utils.two_phase_integrative_model_train_and_evaluate_prevalence_objective(config, None, "todo", print_every=100, trial=trial.number)
    elif args.objective == "three_objectives":
        model, total_nll, train_nll, val_nll, train_loss = optimization_utils.two_phase_integrative_model_train_and_evaluate_three_objectives(config, None, "todo", print_every=100, trial=trial.number)
    else:
        raise ValueError(f"Unknown objective function: {args.objective}")    
    
    trial.set_user_attr("train_nll", float(train_nll))
    trial.set_user_attr("val_nll", float(val_nll))
    trial.set_user_attr("total_nll", float(total_nll))
    trial.set_user_attr("train_loss", float(train_loss))
    # --- save ONLY if this trial is the best so far ---
    try:
        completed = [t for t in trial.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        prev_best = min([t.value for t in completed]) if completed else None
    except Exception:
        prev_best = None
    is_new_best = (prev_best is None) or (float(val_nll) < float(prev_best))
    trial.set_user_attr("is_new_best", bool(is_new_best))

    if is_new_best:
        import equinox as eqx, json, pathlib

        outdir = pathlib.Path(f"{config['data_kwargs']['town']}/optuna_best_{config['phase_cut_date']}_prev{config['prev_phase_cut_date']}_{args.objective}")
        outdir.mkdir(parents=True, exist_ok=True)
        eqx.tree_serialise_leaves(outdir / "model.eqx", model)
        (outdir / "hparams.json").write_text(json.dumps(trial.params, indent=2))
        meta = {
            "trial_number": int(trial.number),
            "train_nll": float(train_nll),
            "val_nll": float(val_nll),
            "total_nll": float(total_nll),
            "train_loss": float(train_loss),
            "timestamp": datetime.now().isoformat()
        }
        (outdir / "metrics.json").write_text(json.dumps(meta, indent=2))
    # --------------------------------------------------

    # jax.clear_caches()
    gc.collect()
    
    return val_nll

pathlib.Path(config["data_kwargs"]["town"]).mkdir(parents=True, exist_ok=True)
storage = f"sqlite:///{config['data_kwargs']['town']}/optuna_study_two_phase_model_{config['phase_cut_date']}_prev{config['prev_phase_cut_date']}_{args.objective}.db?timeout=600&journal_mode=WAL"

study = optuna.create_study(
    study_name="ude_hp_search_cc_ude",
    direction="minimize",
    storage=storage,  # Save to local SQLite DB
    load_if_exists=True)
study.optimize(objective, n_trials=500, gc_after_trial=True, n_jobs=1)

print("Best trial:")
trial = study.best_trial
print(f"  Val NegLL: {trial.value}")
print(f"  Params: {trial.params}")
