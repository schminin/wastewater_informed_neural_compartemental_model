import datetime
from pathlib import Path
from tensorboardX import SummaryWriter
import json

import io
from PIL import Image
import jax
import jax.numpy as jnp

import equinox as eqx 
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jr
import optax  

import matplotlib.pyplot as plt
import time

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import model_definition.model_utils as model_utils
import model_definition.two_phase_integrative_ude as two_phase_integrative_ude
import model_definition.two_phase_integrative_ude_constant_transmission as two_phase_integrative_ude_constant_transmission
import model_definition.two_phase_integrative_ude_no_t_norm as two_phase_integrative_ude_no_t_norm
import data.data_utils as data_utils
import data.data_utils_integrative_model as data_utils_integrative_model

from jax.scipy.special import gammaln
from jax.scipy.special import log_ndtr

import pandas as pd

def define_experiment_logger(config, experiment_series, experiment_name=None):
    # Log Training process
    parent_log_dir = f"results/{experiment_series}"
    Path(parent_log_dir).mkdir(parents=True, exist_ok=True)

    if not experiment_name:
        experiment_name = datetime.datetime.now().strftime('%Y%m%d_%H%M')

    experiment_log_dir = f"{parent_log_dir}/{experiment_name}"
    writer = SummaryWriter(log_dir=experiment_log_dir)
    layout = {
        "train_val_curves": {
            "loss": ["Multiline", ["loss/train", "loss/validation"]],
        },
    }    
    writer.add_custom_scalars(layout)
    # Save static config (e.g., architecture)
    with open(f"{experiment_log_dir}/model_config.json", "w") as f:
        json.dump(config, f)

    print(f"Logging to {experiment_log_dir}")
    return experiment_log_dir, writer



def two_phase_integrative_model_load_data(config):
    dt = config.get("dt", 1)  # days
    assert abs(round(1/dt) - (1/dt)) < 1e-8, f"1/dt must be an integer, got dt={dt} (1/dt={1/dt})"


    y_name, obs_dates, t_scale, obs_day_idx, ys, population_size, prevalence = data_utils_integrative_model.load_data(config["data_kwargs"], t_normalized=False)
    n_days_pred_conc = config.get("n_days_pred_conc", 0)  # days to predict beyond last observed date for concentration
    if n_days_pred_conc > 0:
        # update data used for training - for visualizations, the whole dataset is necessary. 
        criterium = obs_day_idx > (obs_day_idx.max()-n_days_pred_conc)
        test_conc = ys[criterium,0]
        conc_dates_test = obs_dates[criterium]
        ys[criterium,0] = jnp.nan
    else:
        test_conc = None
        conc_dates_test = None

    # day numbering for simulation days
    start_day = int(obs_day_idx.min()) - 1
    end_day   = int(obs_day_idx.max())                # inclusive
    sim_day_idx = jnp.arange(start_day, end_day + dt, step=dt)    # days
    sim_idx     = jnp.arange(len(sim_day_idx), dtype=jnp.int32)
    # time input for neural network, i.e. all simulation timepoints normalized + corresponding dates
    t_sim = (sim_day_idx/t_scale).astype(jnp.float32)
    dates_sim = pd.date_range(pd.to_datetime(obs_dates).min()-pd.Timedelta(days=1), pd.to_datetime(obs_dates).max(), freq=f'{dt}D')

    # training is implemented in a two-phase approach:
    phase_cut_date = config["phase_cut_date"] # "2023-03-14"

    # case count data
    mask_phase_1_obs = pd.to_datetime(obs_dates)<=phase_cut_date
    mask_phase_2_obs = pd.to_datetime(obs_dates)>(pd.Timestamp(phase_cut_date)+pd.Timedelta(days=7)) # 7d sum, so we need to ensure that no values of phase I are included in phase II
    # split to two phases
    obs_dates_phase_1 = obs_dates[mask_phase_1_obs]
    obs_dates_phase_2 = obs_dates[mask_phase_2_obs]
    obs_I_phase_1 = ys[:,1][mask_phase_1_obs]
    obs_I_phase_2 = ys[:,1][mask_phase_2_obs]
    if config["data_kwargs"]["town"] != "Bonn":
        phase_cut_date_prev = config.get("prev_phase_cut_date", phase_cut_date) 
        mask_phase_1_prev = pd.to_datetime(prevalence["Date"])<=phase_cut_date_prev
        mask_phase_2_prev = pd.to_datetime(prevalence["Date"])>phase_cut_date_prev
        obs_n_tests_phase_1 = prevalence.loc[mask_phase_1_prev, "Valid test"].astype(int).values
        obs_n_tests_phase_2 = prevalence.loc[mask_phase_2_prev, "Valid test"].astype(int).values
        obs_n_tests = prevalence["Valid test"].astype(int).values
        obs_pos_tests_phase_1 = prevalence.loc[mask_phase_1_prev, "Positive test"].astype(int).values
        obs_pos_tests_phase_2 = prevalence.loc[mask_phase_2_prev, "Positive test"].astype(int).values
        obs_pos_tests = prevalence["Positive test"].astype(int).values
        obs_dates_phase_1_prev = prevalence.loc[prevalence["Date"]<=phase_cut_date_prev, "Date"].values
        obs_dates_phase_2_prev = prevalence.loc[prevalence["Date"]>phase_cut_date_prev, "Date"].values
        obs_dates_prev = prevalence["Date"].values

    # drop NaNs in observations
    phase_1_nan_mask = jnp.isnan(obs_I_phase_1)
    obs_I_phase_1 = obs_I_phase_1[~phase_1_nan_mask]
    phase_2_nan_mask = jnp.isnan(obs_I_phase_2)
    obs_I_phase_2 = obs_I_phase_2[~phase_2_nan_mask]
    obs_dates_phase_1 = obs_dates_phase_1[~phase_1_nan_mask]
    obs_dates_phase_2 = obs_dates_phase_2[~phase_2_nan_mask]
    obs_dates_I = obs_dates[~jnp.isnan(ys[:,1])]
    I_all = ys[~jnp.isnan(ys[:,1]),1]

    # concentration data
    min_day_for_conc_eval = start_day + int(config["T_max"])
    eval_conc = ys[obs_day_idx>min_day_for_conc_eval,0] # ensure that only dates with appropriate convoluted values are used
    eval_conc_dates = obs_dates[obs_day_idx>min_day_for_conc_eval]
    # drop NaNs
    eval_conc_dates = eval_conc_dates[~jnp.isnan(eval_conc)]
    eval_conc = eval_conc[~jnp.isnan(eval_conc)]

    # train–validation split (random, reproducible)
    val_fraction = float(config.get("val_fraction", 0.125))  # default ~1/8
    key = jr.key(int(config.get("seed", 0)))         # set seed via config

    def random_split_indices(idx, key):
        n = idx.shape[0]
        # ensure at least 1 val point when possible, but never exceed n
        n_val = int(jnp.clip(jnp.round(n * val_fraction), a_min=1 if n > 1 else 0, a_max=n))
        perm = jr.permutation(key, idx)          # shuffle indices
        val_idx = jnp.sort(perm[:n_val])         # keep chronological order for masks
        train_idx = jnp.sort(perm[n_val:])
        return train_idx, val_idx

    # I (incidence) split
    idx_I = jnp.arange(len(obs_I_phase_1))
    train_idx_I, val_idx_I = random_split_indices(idx_I, key)
    key, key_C = jr.split(key)

    I_train, I_val = obs_I_phase_1[train_idx_I], obs_I_phase_1[val_idx_I]
    I_dates_train, I_dates_val = obs_dates_phase_1[train_idx_I], obs_dates_phase_1[val_idx_I]

    # concentration split
    idx_C = jnp.arange(len(eval_conc))
    train_idx_C, val_idx_C = random_split_indices(idx_C, key_C)
    key, key_prev = jr.split(key)

    conc_train, conc_val = eval_conc[train_idx_C], eval_conc[val_idx_C]
    conc_dates_train, conc_dates_val = eval_conc_dates[train_idx_C], eval_conc_dates[val_idx_C]

    # prevalence split (only if available)
    if config["data_kwargs"]["town"] != "Bonn":
        idx_prev = jnp.arange(len(obs_n_tests_phase_1))
        train_idx_prev, val_idx_prev = random_split_indices(idx_prev, key_prev)

        n_tests_train, n_tests_val = obs_n_tests_phase_1[train_idx_prev], obs_n_tests_phase_1[val_idx_prev]
        pos_tests_train, pos_tests_val = obs_pos_tests_phase_1[train_idx_prev], obs_pos_tests_phase_1[val_idx_prev]
        prevalence_dates_train, prevalence_dates_val = obs_dates_phase_1_prev[train_idx_prev], obs_dates_phase_1_prev[val_idx_prev]

        t_mask_prev_train = (sim_idx[dates_sim.isin(prevalence_dates_train)]).astype(jnp.int32)
        t_mask_prev_val   = (sim_idx[dates_sim.isin(prevalence_dates_val)]).astype(jnp.int32)
        t_mask_prev_test = (sim_idx[dates_sim.isin(obs_dates_phase_2_prev)]).astype(jnp.int32) 
        t_mask_prev_all = (sim_idx[dates_sim.isin(obs_dates_prev)]).astype(jnp.int32)
    else:
        n_tests_train, n_tests_val = None, None
        pos_tests_val, pos_tests_train = None, None
        obs_n_tests_phase_2 = None
        obs_pos_tests_phase_2 = None
        obs_n_tests, obs_pos_tests = None, None

        prevalence_dates_train = None
        prevalence_dates_val = None
        obs_dates_phase_2_prev = None
        obs_dates_prev = None

        t_mask_prev_train = None
        t_mask_prev_val = None
        t_mask_prev_test = None
        t_mask_prev_all = None

    # create t values
    t_phase_1 = t_sim[dates_sim <= phase_cut_date]

    # create masks for prediction processing
    t_mask_I_train = (sim_idx[dates_sim.isin(I_dates_train)] - 1).astype(jnp.int32)
    t_mask_I_val   = (sim_idx[dates_sim.isin(I_dates_val)]   - 1).astype(jnp.int32)
    t_mask_I_all   = (sim_idx[dates_sim.isin(obs_dates_I)]   - 1).astype(jnp.int32)
    t_mask_I_test  = (sim_idx[dates_sim.isin(obs_dates_phase_2)] - 1).astype(jnp.int32)

    # create masks for prediction processing
    T_max_steps = int(round(config["T_max"] / dt))
    valid_for_conc = sim_day_idx >= (start_day + config["T_max"])
    sel_train = valid_for_conc & dates_sim.isin(conc_dates_train)
    sel_val   = valid_for_conc & dates_sim.isin(conc_dates_val)
    sel_all   = valid_for_conc & dates_sim.isin(eval_conc_dates)

    t_mask_conc_train = sim_idx[sel_train] - T_max_steps
    t_mask_conc_val   = sim_idx[sel_val]   - T_max_steps
    t_mask_conc_all   = sim_idx[sel_all]   - T_max_steps

    data = {
        "t_all": t_sim,
        "t_all_idx": sim_idx,
        "t_phase_1": t_phase_1,
        "t_mask_I_train": t_mask_I_train,
        "t_mask_I_val": t_mask_I_val,
        "t_mask_I_test": t_mask_I_test,
        "t_mask_I_all": t_mask_I_all,
        "I_train": I_train,
        "I_val": I_val,
        "obs_dates_phase_2": obs_dates_phase_2,
        "I_test": obs_I_phase_2,
        "I_all": I_all,
        "dates_all": dates_sim,
        "I_dates_train": I_dates_train,
        "I_dates_val": I_dates_val,
        # concentration
        "conc_train": conc_train,
        "conc_val": conc_val,
        "eval_conc": eval_conc,
        "t_mask_conc_train": t_mask_conc_train,
        "t_mask_conc_val": t_mask_conc_val,
        "t_mask_conc_all": t_mask_conc_all,
        "conc_dates_train": conc_dates_train,
        "conc_dates_val": conc_dates_val,
        "conc_dates_test": conc_dates_test,
        "eval_conc_dates": eval_conc_dates,
        "test_conc": test_conc,
        # other info
        "population_size": population_size,
        "t_scale": t_scale,
        "town": config["data_kwargs"].get("town", "unknown"),
        # prevalence
        "n_tests_train": n_tests_train,
        "n_tests_val": n_tests_val,
        "n_tests_test": obs_n_tests_phase_2,
        "n_tests": obs_n_tests,
        "pos_tests_train": pos_tests_train,
        "pos_tests_val": pos_tests_val,
        "pos_tests_test": obs_pos_tests_phase_2,
        "pos_tests": obs_pos_tests,
        "prevalence_dates_train": prevalence_dates_train,
        "prevalence_dates_val": prevalence_dates_val,
        "prevalence_dates_test": obs_dates_phase_2_prev,
        "t_mask_prev_train": t_mask_prev_train,
        "t_mask_prev_val": t_mask_prev_val,
        "t_mask_prev_test": t_mask_prev_test,
        "t_mask_prev_all": t_mask_prev_all
    }
    return data



def two_phase_integrative_model_no_t_norm_train_and_evaluate(config, writer, experiment_log_dir, trial=1, print_every=100):
    
    data = two_phase_integrative_model_load_data(config)

    model_key = jr.key(config["seed"])

    model = two_phase_integrative_ude_no_t_norm.IntegrativeModel(
        width_size=config["width_size"],
        depth=config["depth"],
        activation=model_utils.activation_fct_mapper[config["activation"]],
        underreporting_model=config.get("underreporting_model", "default"),
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
        key=model_key,
        init_par_vmr=config.get("init_par_vmr", 1.0),
        init_sigma_C=config.get("init_sigma_C", 1.0),
        reporting_delay=config.get("reporting_delay", 3)  # days, delay between infection and reporting
    )


    @eqx.filter_value_and_grad
    def grad_loss(model, t_all, t_phase_1, t_mask_ids_I, t_mask_ids_conc, obs_cases, obs_conc):
        # Forward pass
        pred_conc, I_new_7d_pred = model(t_all, t_phase_1)

        # Select observed timestamps from predictions
        I_new_7d_sel = I_new_7d_pred[t_mask_ids_I]
        conc_sel     = pred_conc[t_mask_ids_conc]

        # Masks to ignore NaNs from 7-day window/reporting delay and missing obs
        I_valid = (~jnp.isnan(I_new_7d_sel))
        C_valid = (~jnp.isnan(conc_sel))

        pred_I_masked = jnp.where(I_valid, I_new_7d_sel, 0.0)
        obs_I_masked  = jnp.where(I_valid, obs_cases, 0.0)
        pred_C_masked = jnp.where(C_valid, conc_sel, 0.0)
        obs_C_masked  = jnp.where(C_valid, obs_conc, 0.0)

        # Pull σ from the model
        # sigma_I = jnp.exp(model.log_sigma_I)
        sigma_C = jnp.exp(model.log_sigma_C)
        vmr_I = 1.0 + jax.nn.softplus(model.par_vmr)

        I_mask = jnp.sum(jnp.where(I_valid, 1, 0))
        C_mask = jnp.sum(jnp.where(C_valid, 1, 0))

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

        any_inf = jnp.any(jnp.isinf(I_new_7d_sel)) | jnp.any(jnp.isinf(conc_sel))
        any_nan = jnp.isnan(nll_I / jnp.maximum(I_mask, 1)) | jnp.isnan(nll_C / jnp.maximum(C_mask, 1))
        return jnp.where(any_inf | any_nan, jnp.inf, (nll_I / I_mask) + (nll_C / C_mask) + config.get("reg_norm")*model.beta_regularization_loss(ts=t_all, mode=config.get("regularization_mode"))) + config.get("underreporting_reg_norm", 0)*model.underreporting_regularization_loss(ts=t_all, mode=config.get("underreporting_regularization_mode", "None"))

    def plain_negll(model, t_all, t_phase_1, t_mask_ids_I, t_mask_ids_conc, obs_cases, obs_conc):
        # Forward pass
        pred_conc, I_new_7d_pred = model(t_all, t_phase_1)

        # Select observed timestamps from predictions
        I_new_7d_sel = I_new_7d_pred[t_mask_ids_I]
        conc_sel     = pred_conc[t_mask_ids_conc]

        # Masks to ignore NaNs from 7-day window/reporting delay and missing obs
        I_valid = (~jnp.isnan(I_new_7d_sel))
        C_valid = (~jnp.isnan(conc_sel))

        pred_I_masked = jnp.where(I_valid, I_new_7d_sel, 0.0)
        obs_I_masked  = jnp.where(I_valid, obs_cases, 0.0)
        pred_C_masked = jnp.where(C_valid, conc_sel, 0.0)
        obs_C_masked  = jnp.where(C_valid, obs_conc, 0.0)

        # Pull σ from the model
        # sigma_I = jnp.exp(model.log_sigma_I)
        sigma_C = jnp.exp(model.log_sigma_C)
        vmr_I = 1.0 + jax.nn.softplus(model.par_vmr)

        I_mask = jnp.sum(jnp.where(I_valid, 1, 0))
        C_mask = jnp.sum(jnp.where(C_valid, 1, 0))

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

        any_inf = jnp.any(jnp.isinf(I_new_7d_sel)) | jnp.any(jnp.isinf(conc_sel))
        any_nan = jnp.isnan(nll_I / jnp.maximum(I_mask, 1)) | jnp.isnan(nll_C / jnp.maximum(C_mask, 1))
        return jnp.where(any_inf | any_nan, jnp.inf, (nll_I / I_mask) + (nll_C / C_mask)) # + config.get("reg_norm")*model.regularization_loss(ts=t_all, mode=config.get("regularization_mode")))

    @eqx.filter_jit
    def make_step(t_all, t_phase_1, I_train, conc_train, t_mask_I_train, t_mask_conc_train, model, opt_state):
        loss, grads = grad_loss(model, t_all, t_phase_1, t_mask_I_train, t_mask_conc_train, I_train, conc_train)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    for stage, (lr, steps) in enumerate(zip(config["lr_strategy"], config["steps_strategy"])):
        stage_tag = f"stage_{stage+1}"

        if writer:
            print(f"Training {stage_tag} with learning rate {lr} for {steps} steps")

        optim = optax.adabelief(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

        best_val_negll = jnp.inf
        loss = jnp.inf
        best_model = model
        for step in range(steps):
            val_negll = plain_negll(model, data["t_all"], data["t_phase_1"], data["t_mask_I_val"], data["t_mask_conc_val"], data["I_val"], data["conc_val"])
            if val_negll < best_val_negll:
                best_val_negll = val_negll
                best_model = model
                train_loss = loss
            global_step = sum(config["steps_strategy"][:stage]) + step
            loss, model, opt_state = make_step(data["t_all"], data["t_phase_1"], data["I_train"], data["conc_train"], data["t_mask_I_train"], data["t_mask_conc_train"], model, opt_state)

            if writer:
                writer.add_scalar(f"loss/train", loss.item(), global_step)
                writer.add_scalar(f"loss/val", val_negll.item(), global_step)
                writer.add_scalar(f"MechanisticParameter/par_vmr", model.par_vmr.item(), global_step)
                writer.add_scalar(f"MechanisticParameter/log_sigma_C", model.log_sigma_C.item(), global_step)
                if (step % print_every) == 0 or step == steps - 1:
                    print(f"Step: {step}, Loss: {loss}")

            # Single guard: restore previous good model on NaN/Inf and break
            if jnp.isnan(loss) or jnp.isinf(loss):
                if writer:
                    print(f"✱ Caught invalid loss at stage {stage}, step {step}; restoring previous model and breaking.")
                break

    # Final training loss (same masking as during training)
    train_nll = plain_negll(best_model, data["t_all"], data["t_phase_1"], data["t_mask_I_train"], data["t_mask_conc_train"], data["I_train"], data["conc_train"])
    val_nll = plain_negll(best_model, data["t_all"], data["t_phase_1"], data["t_mask_I_val"], data["t_mask_conc_val"], data["I_val"], data["conc_val"])
    total_nll = plain_negll(best_model, data["t_all"], data["t_phase_1"], data["t_mask_I_all"], data["t_mask_conc_all"], data["I_all"], data["eval_conc"])
    return best_model, total_nll, train_nll, val_nll, train_loss



def two_phase_integrative_model_constant_transmission_train_and_evaluate(config, writer, experiment_log_dir, trial=1, print_every=100):
    
    data = two_phase_integrative_model_load_data(config)

    model_key = jr.key(config["seed"])

    model = two_phase_integrative_ude_constant_transmission.IntegrativeModel(
        width_size=config["width_size"],
        depth=config["depth"],
        activation=model_utils.activation_fct_mapper[config["activation"]],
        underreporting_model=config.get("underreporting_model", "default"),
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
        key=model_key,
        init_par_vmr=config.get("init_par_vmr", 1.0),
        init_sigma_C=config.get("init_sigma_C", 1.0),
        reporting_delay=config.get("reporting_delay", 3),
        initial_transmission=config.get("initial_transmission", 0.3),
    )


    @eqx.filter_value_and_grad
    def grad_loss(model, t_all, t_phase_1, t_mask_ids_I, t_mask_ids_conc, obs_cases, obs_conc):
        # Forward pass
        pred_conc, I_new_7d_pred = model(t_all, t_phase_1)

        # Select observed timestamps from predictions
        I_new_7d_sel = I_new_7d_pred[t_mask_ids_I]
        conc_sel     = pred_conc[t_mask_ids_conc]

        # Masks to ignore NaNs from 7-day window/reporting delay and missing obs
        I_valid = (~jnp.isnan(I_new_7d_sel))
        C_valid = (~jnp.isnan(conc_sel))

        pred_I_masked = jnp.where(I_valid, I_new_7d_sel, 0.0)
        obs_I_masked  = jnp.where(I_valid, obs_cases, 0.0)
        pred_C_masked = jnp.where(C_valid, conc_sel, 0.0)
        obs_C_masked  = jnp.where(C_valid, obs_conc, 0.0)

        # Pull σ from the model
        # sigma_I = jnp.exp(model.log_sigma_I)
        sigma_C = jnp.exp(model.log_sigma_C)
        vmr_I = 1.0 + jax.nn.softplus(model.par_vmr)

        I_mask = jnp.sum(jnp.where(I_valid, 1, 0))
        C_mask = jnp.sum(jnp.where(C_valid, 1, 0))

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

        any_inf = jnp.any(jnp.isinf(I_new_7d_sel)) | jnp.any(jnp.isinf(conc_sel))
        any_nan = jnp.isnan(nll_I / jnp.maximum(I_mask, 1)) | jnp.isnan(nll_C / jnp.maximum(C_mask, 1))
        return jnp.where(any_inf | any_nan, jnp.inf, (nll_I / I_mask) + (nll_C / C_mask) + config.get("reg_norm")*model.beta_regularization_loss(ts=t_all, mode=config.get("regularization_mode"))) + config.get("underreporting_reg_norm", 0)*model.underreporting_regularization_loss(ts=t_all, mode=config.get("underreporting_regularization_mode", "None"))

    def plain_negll(model, t_all, t_phase_1, t_mask_ids_I, t_mask_ids_conc, obs_cases, obs_conc):
        # Forward pass
        pred_conc, I_new_7d_pred = model(t_all, t_phase_1)

        # Select observed timestamps from predictions
        I_new_7d_sel = I_new_7d_pred[t_mask_ids_I]
        conc_sel     = pred_conc[t_mask_ids_conc]

        # Masks to ignore NaNs from 7-day window/reporting delay and missing obs
        I_valid = (~jnp.isnan(I_new_7d_sel))
        C_valid = (~jnp.isnan(conc_sel))

        pred_I_masked = jnp.where(I_valid, I_new_7d_sel, 0.0)
        obs_I_masked  = jnp.where(I_valid, obs_cases, 0.0)
        pred_C_masked = jnp.where(C_valid, conc_sel, 0.0)
        obs_C_masked  = jnp.where(C_valid, obs_conc, 0.0)

        # Pull σ from the model
        # sigma_I = jnp.exp(model.log_sigma_I)
        sigma_C = jnp.exp(model.log_sigma_C)
        vmr_I = 1.0 + jax.nn.softplus(model.par_vmr)

        I_mask = jnp.sum(jnp.where(I_valid, 1, 0))
        C_mask = jnp.sum(jnp.where(C_valid, 1, 0))

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

        any_inf = jnp.any(jnp.isinf(I_new_7d_sel)) | jnp.any(jnp.isinf(conc_sel))
        any_nan = jnp.isnan(nll_I / jnp.maximum(I_mask, 1)) | jnp.isnan(nll_C / jnp.maximum(C_mask, 1))
        return jnp.where(any_inf | any_nan, jnp.inf, (nll_I / I_mask) + (nll_C / C_mask)) # + config.get("reg_norm")*model.regularization_loss(ts=t_all, mode=config.get("regularization_mode")))

    @eqx.filter_jit
    def make_step(t_all, t_phase_1, I_train, conc_train, t_mask_I_train, t_mask_conc_train, model, opt_state):
        loss, grads = grad_loss(model, t_all, t_phase_1, t_mask_I_train, t_mask_conc_train, I_train, conc_train)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    for stage, (lr, steps) in enumerate(zip(config["lr_strategy"], config["steps_strategy"])):
        stage_tag = f"stage_{stage+1}"

        if writer:
            print(f"Training {stage_tag} with learning rate {lr} for {steps} steps")

        optim = optax.adabelief(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

        best_val_negll = jnp.inf
        loss = jnp.inf
        best_model = model
        for step in range(steps):
            val_negll = plain_negll(model, data["t_all"], data["t_phase_1"], data["t_mask_I_val"], data["t_mask_conc_val"], data["I_val"], data["conc_val"])
            if val_negll < best_val_negll:
                best_val_negll = val_negll
                best_model = model
                train_loss = loss
            global_step = sum(config["steps_strategy"][:stage]) + step
            loss, model, opt_state = make_step(data["t_all"], data["t_phase_1"], data["I_train"], data["conc_train"], data["t_mask_I_train"], data["t_mask_conc_train"], model, opt_state)

            if writer:
                writer.add_scalar(f"loss/train", loss.item(), global_step)
                writer.add_scalar(f"loss/val", val_negll.item(), global_step)
                writer.add_scalar(f"MechanisticParameter/par_vmr", model.par_vmr.item(), global_step)
                writer.add_scalar(f"MechanisticParameter/log_sigma_C", model.log_sigma_C.item(), global_step)
                if (step % print_every) == 0 or step == steps - 1:
                    print(f"Step: {step}, Loss: {loss}")

            # Single guard: restore previous good model on NaN/Inf and break
            if jnp.isnan(loss) or jnp.isinf(loss):
                if writer:
                    print(f"✱ Caught invalid loss at stage {stage}, step {step}; restoring previous model and breaking.")
                break

    # Final training loss (same masking as during training)
    train_nll = plain_negll(best_model, data["t_all"], data["t_phase_1"], data["t_mask_I_train"], data["t_mask_conc_train"], data["I_train"], data["conc_train"])
    val_nll = plain_negll(best_model, data["t_all"], data["t_phase_1"], data["t_mask_I_val"], data["t_mask_conc_val"], data["I_val"], data["conc_val"])
    total_nll = plain_negll(best_model, data["t_all"], data["t_phase_1"], data["t_mask_I_all"], data["t_mask_conc_all"], data["I_all"], data["eval_conc"])
    return best_model, total_nll, train_nll, val_nll, train_loss



def two_phase_integrative_model_train_and_evaluate_cases_only(config, writer, experiment_log_dir, trial=1, print_every=100):
    
    data = two_phase_integrative_model_load_data(config)

    model_key = jr.key(config["seed"])

    model = two_phase_integrative_ude.IntegrativeModel(
        width_size=config["width_size"],
        depth=config["depth"],
        activation=model_utils.activation_fct_mapper[config["activation"]],
        underreporting_model=config.get("underreporting_model", "default"),
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
        key=model_key,
        init_par_vmr=config.get("init_par_vmr", 1.0),
        init_sigma_C=config.get("init_sigma_C", 1.0),
        reporting_delay=config.get("reporting_delay", 3)  # days, delay between infection and reporting
    )


    @eqx.filter_value_and_grad
    def grad_loss(model, t_all, t_phase_1, t_mask_ids_I, t_mask_ids_conc, obs_cases, obs_conc):
        # Forward pass
        pred_conc, I_new_7d_pred = model(t_all, t_phase_1)

        # Select observed timestamps from predictions
        I_new_7d_sel = I_new_7d_pred[t_mask_ids_I]

        # Masks to ignore NaNs from 7-day window/reporting delay and missing obs
        I_valid = (~jnp.isnan(I_new_7d_sel))

        pred_I_masked = jnp.where(I_valid, I_new_7d_sel, 0.0)
        obs_I_masked  = jnp.where(I_valid, obs_cases, 0.0)

        # Pull σ from the model
        # sigma_I = jnp.exp(model.log_sigma_I)
        vmr_I = 1.0 + jax.nn.softplus(model.par_vmr)

        I_mask = jnp.sum(jnp.where(I_valid, 1, 0))

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

        any_inf = jnp.any(jnp.isinf(I_new_7d_sel)) 
        any_nan = jnp.isnan(nll_I / jnp.maximum(I_mask, 1)) 
        return jnp.where(any_inf | any_nan, jnp.inf, (nll_I / I_mask) + config.get("reg_norm")*model.beta_regularization_loss(ts=t_all, mode=config.get("regularization_mode"))) + config.get("underreporting_reg_norm", 0)*model.underreporting_regularization_loss(ts=t_all, mode=config.get("underreporting_regularization_mode", "None"))

    def plain_negll(model, t_all, t_phase_1, t_mask_ids_I, t_mask_ids_conc, obs_cases, obs_conc):
        # Forward pass
        pred_conc, I_new_7d_pred = model(t_all, t_phase_1)

        # Select observed timestamps from predictions
        I_new_7d_sel = I_new_7d_pred[t_mask_ids_I]

        # Masks to ignore NaNs from 7-day window/reporting delay and missing obs
        I_valid = (~jnp.isnan(I_new_7d_sel))

        pred_I_masked = jnp.where(I_valid, I_new_7d_sel, 0.0)
        obs_I_masked  = jnp.where(I_valid, obs_cases, 0.0)

        # Pull σ from the model
        # sigma_I = jnp.exp(model.log_sigma_I)
        sigma_C = jnp.exp(model.log_sigma_C)
        vmr_I = 1.0 + jax.nn.softplus(model.par_vmr)

        I_mask = jnp.sum(jnp.where(I_valid, 1, 0))

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

        any_inf = jnp.any(jnp.isinf(I_new_7d_sel))
        any_nan = jnp.isnan(nll_I / jnp.maximum(I_mask, 1))
        return jnp.where(any_inf | any_nan, jnp.inf, (nll_I / I_mask)) # + config.get("reg_norm")*model.regularization_loss(ts=t_all, mode=config.get("regularization_mode")))

    @eqx.filter_jit
    def make_step(t_all, t_phase_1, I_train, conc_train, t_mask_I_train, t_mask_conc_train, model, opt_state):
        loss, grads = grad_loss(model, t_all, t_phase_1, t_mask_I_train, t_mask_conc_train, I_train, conc_train)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    for stage, (lr, steps) in enumerate(zip(config["lr_strategy"], config["steps_strategy"])):
        stage_tag = f"stage_{stage+1}"

        if writer:
            print(f"Training {stage_tag} with learning rate {lr} for {steps} steps")

        optim = optax.adabelief(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

        best_val_negll = jnp.inf
        loss = jnp.inf
        best_model = model
        for step in range(steps):
            val_negll = plain_negll(model, data["t_all"], data["t_phase_1"], data["t_mask_I_val"], data["t_mask_conc_val"], data["I_val"], data["conc_val"])
            if val_negll < best_val_negll:
                best_val_negll = val_negll
                best_model = model
                train_loss = loss
            global_step = sum(config["steps_strategy"][:stage]) + step
            loss, model, opt_state = make_step(data["t_all"], data["t_phase_1"], data["I_train"], data["conc_train"], data["t_mask_I_train"], data["t_mask_conc_train"], model, opt_state)

            # Single guard: restore previous good model on NaN/Inf and break
            if jnp.isnan(loss) or jnp.isinf(loss):
                if writer:
                    print(f"✱ Caught invalid loss at stage {stage}, step {step}; restoring previous model and breaking.")
                break

    # Final training loss (same masking as during training)
    train_nll = plain_negll(best_model, data["t_all"], data["t_phase_1"], data["t_mask_I_train"], data["t_mask_conc_train"], data["I_train"], data["conc_train"])
    val_nll = plain_negll(best_model, data["t_all"], data["t_phase_1"], data["t_mask_I_val"], data["t_mask_conc_val"], data["I_val"], data["conc_val"])
    total_nll = plain_negll(best_model, data["t_all"], data["t_phase_1"], data["t_mask_I_all"], data["t_mask_conc_all"], data["I_all"], data["eval_conc"])
    return best_model, total_nll, train_nll, val_nll, train_loss




def two_phase_integrative_model_train_and_evaluate(config, writer, experiment_log_dir, trial=1, print_every=100):
    
    data = two_phase_integrative_model_load_data(config)

    model_key = jr.key(config["seed"])

    model = two_phase_integrative_ude.IntegrativeModel(
        width_size=config["width_size"],
        depth=config["depth"],
        activation=model_utils.activation_fct_mapper[config["activation"]],
        underreporting_model=config.get("underreporting_model", "default"),
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
        key=model_key,
        init_par_vmr=config.get("init_par_vmr", 1.0),
        init_sigma_C=config.get("init_sigma_C", 1.0),
        reporting_delay=config.get("reporting_delay", 3)  # days, delay between infection and reporting
    )


    @eqx.filter_value_and_grad
    def grad_loss(model, t_all, t_phase_1, t_mask_ids_I, t_mask_ids_conc, obs_cases, obs_conc):
        # Forward pass
        pred_conc, I_new_7d_pred = model(t_all, t_phase_1)

        # Select observed timestamps from predictions
        I_new_7d_sel = I_new_7d_pred[t_mask_ids_I]
        conc_sel     = pred_conc[t_mask_ids_conc]

        # Masks to ignore NaNs from 7-day window/reporting delay and missing obs
        I_valid = (~jnp.isnan(I_new_7d_sel))
        C_valid = (~jnp.isnan(conc_sel))

        pred_I_masked = jnp.where(I_valid, I_new_7d_sel, 0.0)
        obs_I_masked  = jnp.where(I_valid, obs_cases, 0.0)
        pred_C_masked = jnp.where(C_valid, conc_sel, 0.0)
        obs_C_masked  = jnp.where(C_valid, obs_conc, 0.0)

        # Pull σ from the model
        # sigma_I = jnp.exp(model.log_sigma_I)
        sigma_C = jnp.exp(model.log_sigma_C)
        vmr_I = 1.0 + jax.nn.softplus(model.par_vmr)

        I_mask = jnp.sum(jnp.where(I_valid, 1, 0))
        C_mask = jnp.sum(jnp.where(C_valid, 1, 0))

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

        any_inf = jnp.any(jnp.isinf(I_new_7d_sel)) | jnp.any(jnp.isinf(conc_sel))
        any_nan = jnp.isnan(nll_I / jnp.maximum(I_mask, 1)) | jnp.isnan(nll_C / jnp.maximum(C_mask, 1))
        return jnp.where(any_inf | any_nan, jnp.inf, (nll_I / I_mask) + (nll_C / C_mask) + config.get("reg_norm")*model.beta_regularization_loss(ts=t_all, mode=config.get("regularization_mode"))) + config.get("underreporting_reg_norm", 0)*model.underreporting_regularization_loss(ts=t_all, mode=config.get("underreporting_regularization_mode", "None"))

    def plain_negll(model, t_all, t_phase_1, t_mask_ids_I, t_mask_ids_conc, obs_cases, obs_conc):
        # Forward pass
        pred_conc, I_new_7d_pred = model(t_all, t_phase_1)

        # Select observed timestamps from predictions
        I_new_7d_sel = I_new_7d_pred[t_mask_ids_I]
        conc_sel     = pred_conc[t_mask_ids_conc]

        # Masks to ignore NaNs from 7-day window/reporting delay and missing obs
        I_valid = (~jnp.isnan(I_new_7d_sel))
        C_valid = (~jnp.isnan(conc_sel))

        pred_I_masked = jnp.where(I_valid, I_new_7d_sel, 0.0)
        obs_I_masked  = jnp.where(I_valid, obs_cases, 0.0)
        pred_C_masked = jnp.where(C_valid, conc_sel, 0.0)
        obs_C_masked  = jnp.where(C_valid, obs_conc, 0.0)

        # Pull σ from the model
        # sigma_I = jnp.exp(model.log_sigma_I)
        sigma_C = jnp.exp(model.log_sigma_C)
        vmr_I = 1.0 + jax.nn.softplus(model.par_vmr)

        I_mask = jnp.sum(jnp.where(I_valid, 1, 0))
        C_mask = jnp.sum(jnp.where(C_valid, 1, 0))

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

        any_inf = jnp.any(jnp.isinf(I_new_7d_sel)) | jnp.any(jnp.isinf(conc_sel))
        any_nan = jnp.isnan(nll_I / jnp.maximum(I_mask, 1)) | jnp.isnan(nll_C / jnp.maximum(C_mask, 1))
        return jnp.where(any_inf | any_nan, jnp.inf, (nll_I / I_mask) + (nll_C / C_mask)) # + config.get("reg_norm")*model.regularization_loss(ts=t_all, mode=config.get("regularization_mode")))

    @eqx.filter_jit
    def make_step(t_all, t_phase_1, I_train, conc_train, t_mask_I_train, t_mask_conc_train, model, opt_state):
        loss, grads = grad_loss(model, t_all, t_phase_1, t_mask_I_train, t_mask_conc_train, I_train, conc_train)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    for stage, (lr, steps) in enumerate(zip(config["lr_strategy"], config["steps_strategy"])):
        stage_tag = f"stage_{stage+1}"

        if writer:
            print(f"Training {stage_tag} with learning rate {lr} for {steps} steps")

        optim = optax.adabelief(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

        best_val_negll = jnp.inf
        loss = jnp.inf
        best_model = model
        for step in range(steps):
            val_negll = plain_negll(model, data["t_all"], data["t_phase_1"], data["t_mask_I_val"], data["t_mask_conc_val"], data["I_val"], data["conc_val"])
            if val_negll < best_val_negll:
                best_val_negll = val_negll
                best_model = model
                train_loss = loss
            global_step = sum(config["steps_strategy"][:stage]) + step
            loss, model, opt_state = make_step(data["t_all"], data["t_phase_1"], data["I_train"], data["conc_train"], data["t_mask_I_train"], data["t_mask_conc_train"], model, opt_state)

            if writer:
                writer.add_scalar(f"loss/train", loss.item(), global_step)
                writer.add_scalar(f"loss/val", val_negll.item(), global_step)
                writer.add_scalar(f"MechanisticParameter/par_vmr", model.par_vmr.item(), global_step)
                writer.add_scalar(f"MechanisticParameter/log_sigma_C", model.log_sigma_C.item(), global_step)
                if (step % print_every) == 0 or step == steps - 1:
                    print(f"Step: {step}, Loss: {loss}")

            # Single guard: restore previous good model on NaN/Inf and break
            if jnp.isnan(loss) or jnp.isinf(loss):
                if writer:
                    print(f"✱ Caught invalid loss at stage {stage}, step {step}; restoring previous model and breaking.")
                break

    # Final training loss (same masking as during training)
    train_nll = plain_negll(best_model, data["t_all"], data["t_phase_1"], data["t_mask_I_train"], data["t_mask_conc_train"], data["I_train"], data["conc_train"])
    val_nll = plain_negll(best_model, data["t_all"], data["t_phase_1"], data["t_mask_I_val"], data["t_mask_conc_val"], data["I_val"], data["conc_val"])
    total_nll = plain_negll(best_model, data["t_all"], data["t_phase_1"], data["t_mask_I_all"], data["t_mask_conc_all"], data["I_all"], data["eval_conc"])
    return best_model, total_nll, train_nll, val_nll, train_loss



def two_phase_integrative_model_train_and_evaluate_three_objectives(config, writer, experiment_log_dir, trial=1, print_every=100):
    
    data = two_phase_integrative_model_load_data(config)

    model_key = jr.key(config["seed"])

    model = two_phase_integrative_ude.IntegrativeModel(
        width_size=config["width_size"],
        depth=config["depth"],
        activation=model_utils.activation_fct_mapper[config["activation"]],
        underreporting_model=config.get("underreporting_model", "default"),
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
        key=model_key,
        init_par_vmr=config.get("init_par_vmr", 1.0),
        init_sigma_C=config.get("init_sigma_C", 1.0),
        reporting_delay=config.get("reporting_delay", 3)  # days, delay between infection and reporting
    )


    @eqx.filter_value_and_grad
    def grad_loss(model, t_all, t_phase_1, t_mask_ids_I, t_mask_ids_conc, t_mask_ids_prev, obs_cases, obs_conc, obs_pos_tests, obs_total_tests):
        # Forward pass
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

        any_inf = jnp.any(jnp.isinf(I_new_7d_sel)) | jnp.any(jnp.isinf(conc_sel))
        any_nan = jnp.isnan(nll_I / jnp.maximum(I_mask, 1)) | jnp.isnan(nll_C / jnp.maximum(C_mask, 1))
        return jnp.where(any_inf | any_nan, jnp.inf, (nll_I / I_mask) + (nll_C / C_mask) + nll_prev + config.get("reg_norm")*model.beta_regularization_loss(ts=t_all, mode=config.get("regularization_mode"))) + config.get("underreporting_reg_norm", 0)*model.underreporting_regularization_loss(ts=t_all, mode=config.get("underreporting_regularization_mode", "None"))

    def plain_negll(model, t_all, t_phase_1, t_mask_ids_I, t_mask_ids_conc, t_mask_ids_prev, obs_cases, obs_conc, obs_pos_tests, obs_total_tests):
#        # Forward pass
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

        any_inf = jnp.any(jnp.isinf(I_new_7d_sel)) | jnp.any(jnp.isinf(conc_sel))
        any_nan = jnp.isnan(nll_I / jnp.maximum(I_mask, 1)) | jnp.isnan(nll_C / jnp.maximum(C_mask, 1))
        return jnp.where(any_inf | any_nan, jnp.inf, (nll_I / I_mask) + (nll_C / C_mask) + nll_prev) # + config.get("reg_norm")*model.regularization_loss(ts=t_all, mode=config.get("regularization_mode")))

    @eqx.filter_jit
    def make_step(t_all, t_phase_1, I_train, conc_train, t_mask_I_train, t_mask_conc_train, t_mask_prev_train, pos_tests_train, total_tests_train, model, opt_state):
        loss, grads = grad_loss(model, t_all, t_phase_1, t_mask_I_train, t_mask_conc_train, t_mask_prev_train, I_train, conc_train, pos_tests_train, total_tests_train)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    for stage, (lr, steps) in enumerate(zip(config["lr_strategy"], config["steps_strategy"])):
        stage_tag = f"stage_{stage+1}"

        if writer:
            print(f"Training {stage_tag} with learning rate {lr} for {steps} steps")

        optim = optax.adabelief(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

        best_val_negll = jnp.inf
        loss = jnp.inf
        best_model = model
        for step in range(steps):
            val_negll = plain_negll(model, data["t_all"], data["t_phase_1"], data["t_mask_I_val"], data["t_mask_conc_val"],data["t_mask_prev_train"], data["I_val"], data["conc_val"], data["pos_tests_train"], data["n_tests_train"])
            if val_negll < best_val_negll:
                best_val_negll = val_negll
                best_model = model
                train_loss = loss
            global_step = sum(config["steps_strategy"][:stage]) + step
            loss, model, opt_state = make_step(data["t_all"], data["t_phase_1"], data["I_train"], data["conc_train"], data["t_mask_I_train"], data["t_mask_conc_train"], data["t_mask_prev_train"], data["pos_tests_train"], data["n_tests_train"], model, opt_state)

            if writer:
                writer.add_scalar(f"loss/train", loss.item(), global_step)
                writer.add_scalar(f"loss/val", val_negll.item(), global_step)
                writer.add_scalar(f"MechanisticParameter/par_vmr", model.par_vmr.item(), global_step)
                writer.add_scalar(f"MechanisticParameter/log_sigma_C", model.log_sigma_C.item(), global_step)
                if (step % print_every) == 0 or step == steps - 1:
                    print(f"Step: {step}, Loss: {loss}")

            # Single guard: restore previous good model on NaN/Inf and break
            if jnp.isnan(loss) or jnp.isinf(loss):
                if writer:
                    print(f"✱ Caught invalid loss at stage {stage}, step {step}; restoring previous model and breaking.")
                break

    # Final training loss (same masking as during training)
    train_nll = plain_negll(model, data["t_all"], data["t_phase_1"], data["t_mask_I_train"], data["t_mask_conc_train"],data["t_mask_prev_train"], data["I_train"], data["conc_train"], data["pos_tests_train"], data["n_tests_train"])
    val_nll = plain_negll(model, data["t_all"], data["t_phase_1"], data["t_mask_I_val"], data["t_mask_conc_val"],data["t_mask_prev_val"], data["I_val"], data["conc_val"], data["pos_tests_val"], data["n_tests_val"])
    total_nll = plain_negll(model, data["t_all"], data["t_phase_1"], data["t_mask_I_all"], data["t_mask_conc_all"],data["t_mask_prev_all"], data["I_all"], data["eval_conc"], data["pos_tests"], data["n_tests"])
    return best_model, val_nll, train_nll, total_nll, train_loss


def two_phase_integrative_model_train_and_evaluate_prevalence_objective(config, writer, experiment_log_dir, trial=1, print_every=100):
    
    data = two_phase_integrative_model_load_data(config)

    model_key = jr.key(config["seed"])

    model = two_phase_integrative_ude.IntegrativeModel(
        width_size=config["width_size"],
        depth=config["depth"],
        activation=model_utils.activation_fct_mapper[config["activation"]],
        underreporting_model=config.get("underreporting_model", "default"),
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
        key=model_key,
        init_par_vmr=config.get("init_par_vmr", 1.0),
        init_sigma_C=config.get("init_sigma_C", 1.0),
        reporting_delay=config.get("reporting_delay", 3)  # days, delay between infection and reporting
    )


    @eqx.filter_value_and_grad
    def grad_loss(model, t_all, t_phase_1, t_mask_ids_conc, t_mask_ids_prev, obs_conc, obs_pos_tests, obs_total_tests):
        # Forward pass
        pred_conc, _, prevalence = model.run_model_with_prevalence_output(t_all, t_phase_1)

        # Select observed timestamps from predictions
        conc_sel     = pred_conc[t_mask_ids_conc]
        prev_sel     = prevalence[t_mask_ids_prev]

        # Masks to ignore NaNs from 7-day window/reporting delay and missing obs
        C_valid = (~jnp.isnan(conc_sel))
        P_valid = (~jnp.isnan(prev_sel))

        pred_C_masked = jnp.where(C_valid, conc_sel, 0.0)
        obs_C_masked  = jnp.where(C_valid, obs_conc, 0.0)
        pred_P_masked = jnp.where(P_valid, prev_sel, 0.0)

        # Pull σ from the model
        sigma_C = jnp.exp(model.log_sigma_C)

        C_mask = jnp.sum(jnp.where(C_valid, 1, 0))
        P_mask = jnp.sum(jnp.where(P_valid, 1, 0))

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

        any_inf = jnp.any(jnp.isinf(conc_sel))
        any_nan = jnp.isnan(nll_C / jnp.maximum(C_mask, 1))
        return jnp.where(any_inf | any_nan, jnp.inf, (nll_C / C_mask) + nll_prev + config.get("reg_norm")*model.beta_regularization_loss(ts=t_all, mode=config.get("regularization_mode"))) + config.get("underreporting_reg_norm", 0)*model.underreporting_regularization_loss(ts=t_all, mode=config.get("underreporting_regularization_mode", "None"))

    def plain_negll(model, t_all, t_phase_1, t_mask_ids_conc, t_mask_ids_prev, obs_conc, obs_pos_tests, obs_total_tests):
        # Forward pass
        pred_conc, _, prevalence = model.run_model_with_prevalence_output(t_all, t_phase_1)

        # Select observed timestamps from predictions
        conc_sel     = pred_conc[t_mask_ids_conc]
        prev_sel     = prevalence[t_mask_ids_prev]

        # Masks to ignore NaNs from 7-day window/reporting delay and missing obs
        C_valid = (~jnp.isnan(conc_sel))
        P_valid = (~jnp.isnan(prev_sel))

        pred_C_masked = jnp.where(C_valid, conc_sel, 0.0)
        obs_C_masked  = jnp.where(C_valid, obs_conc, 0.0)
        pred_P_masked = jnp.where(P_valid, prev_sel, 0.0)

        # Pull σ from the model
        sigma_C = jnp.exp(model.log_sigma_C)

        C_mask = jnp.sum(jnp.where(C_valid, 1, 0))
        P_mask = jnp.sum(jnp.where(P_valid, 1, 0))

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

        any_inf = jnp.any(jnp.isinf(conc_sel))
        any_nan = jnp.isnan(nll_C / jnp.maximum(C_mask, 1))
        return jnp.where(any_inf | any_nan, jnp.inf,(nll_C / C_mask) + nll_prev) # + config.get("reg_norm")*model.regularization_loss(ts=t_all, mode=config.get("regularization_mode")))

    @eqx.filter_jit
    def make_step(t_all, t_phase_1, conc_train, t_mask_conc_train, t_mask_prev_train, pos_tests_train, total_tests_train, model, opt_state):
        loss, grads = grad_loss(model, t_all, t_phase_1, t_mask_conc_train, t_mask_prev_train, conc_train, pos_tests_train, total_tests_train)
        updates, opt_state = optim.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    for stage, (lr, steps) in enumerate(zip(config["lr_strategy"], config["steps_strategy"])):
        stage_tag = f"stage_{stage+1}"

        if writer:
            print(f"Training {stage_tag} with learning rate {lr} for {steps} steps")

        optim = optax.adabelief(lr)
        opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

        best_val_negll = jnp.inf
        loss = jnp.inf
        best_model = model
        for step in range(steps):
            val_negll = plain_negll(model, data["t_all"], data["t_phase_1"], data["t_mask_conc_val"],data["t_mask_prev_train"], data["conc_val"], data["pos_tests_train"], data["n_tests_train"])
            if val_negll < best_val_negll:
                best_val_negll = val_negll
                best_model = model
                train_loss = loss
            global_step = sum(config["steps_strategy"][:stage]) + step
            loss, model, opt_state = make_step(data["t_all"], data["t_phase_1"], data["conc_train"], data["t_mask_conc_train"], data["t_mask_prev_train"], data["pos_tests_train"], data["n_tests_train"], model, opt_state)

            if writer:
                writer.add_scalar(f"loss/train", loss.item(), global_step)
                writer.add_scalar(f"loss/val", val_negll.item(), global_step)
                writer.add_scalar(f"MechanisticParameter/par_vmr", model.par_vmr.item(), global_step)
                writer.add_scalar(f"MechanisticParameter/log_sigma_C", model.log_sigma_C.item(), global_step)
                if (step % print_every) == 0 or step == steps - 1:
                    print(f"Step: {step}, Loss: {loss}")

            # Single guard: restore previous good model on NaN/Inf and break
            if jnp.isnan(loss) or jnp.isinf(loss):
                if writer:
                    print(f"✱ Caught invalid loss at stage {stage}, step {step}; restoring previous model and breaking.")
                break

    # Final training loss (same masking as during training)
    train_nll = plain_negll(model, data["t_all"], data["t_phase_1"], data["t_mask_conc_train"],data["t_mask_prev_train"], data["conc_train"], data["pos_tests_train"], data["n_tests_train"])
    val_nll = plain_negll(model, data["t_all"], data["t_phase_1"], data["t_mask_conc_val"],data["t_mask_prev_val"], data["conc_val"], data["pos_tests_val"], data["n_tests_val"])
    total_nll = plain_negll(model, data["t_all"], data["t_phase_1"], data["t_mask_conc_all"],data["t_mask_prev_all"], data["eval_conc"], data["pos_tests"], data["n_tests"])
    return best_model, val_nll, train_nll, total_nll, train_loss

