
import jax.random as jr
import jax.numpy as jnp
import jax
import equinox as eqx

import json
import sys, os

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..', '..')))
from optimization import optimization_utils

import model_definition.two_phase_integrative_ude as two_phase_integrative_ude
import data.data_utils_integrative_model as data_utils_integrative_model
import model_definition.model_utils as model_utils

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import argparse

_parser = argparse.ArgumentParser(
    description="Run eval_multistart pipeline with configurable cut date and cutoff value."
)
_parser = argparse.ArgumentParser(description="Run two-phase integrative UDE Optuna optimization.")
_parser.add_argument("--phase_cut_date", type=str, required=True, help="Date to cut phases, format YYYY-MM-DD")
_parser.add_argument("--town", type=str, required=True, help="Town name")
_parser.add_argument("--prev_phase_cut_date", type=str, required=True, help="Date to cut phases, format YYYY-MM-DD")
_parser.add_argument("--objective", type=str, required=True, help="Objective function to optimize")
_parser.add_argument("--cutoff_value",required=True,type=float,help="Numeric cutoff value.")

_args = _parser.parse_args()
phase_cut_date = _args.phase_cut_date
cutoff_value = _args.cutoff_value
prev_phase_cut_date = _args.prev_phase_cut_date
town = _args.town
objective = _args.objective

# read in results from multistart
multistart_path = f"{town}/multistart_models/{phase_cut_date}_prev{prev_phase_cut_date}_{objective}"
hparams_path = f"{town}/optuna_best_{phase_cut_date}_prev{prev_phase_cut_date}_{objective}/hparams.json"
out_dir = f"{town}/multistart_results/{phase_cut_date}_prev{prev_phase_cut_date}_{objective}/visualizations_{cutoff_value}"
os.makedirs(out_dir, exist_ok=True)

df = pd.read_csv(f"{town}/multistart_results/{phase_cut_date}_prev{prev_phase_cut_date}_{objective}/multistart_metrics_{phase_cut_date}.csv")


len_pre = len(df)
df = df.loc[df.k1.notna()]
len_post = len(df)

# calculate train/val negll
if objective == "cases_and_conc":
    train_val_negll_c = (df["train_negll_c"]*df["n_obs_train_c"] + df["val_negll_c"]*df["n_obs_val_c"])/(df["n_obs_train_c"] + df["n_obs_val_c"])
    train_val_negll_I = (df["train_negll_I"]*df["n_obs_train_I"] + df["val_negll_I"]*df["n_obs_val_I"])/(df["n_obs_train_I"] + df["n_obs_val_I"])
    df.loc[:,"train_val_negll"] = train_val_negll_c + train_val_negll_I
    df_sub = df.loc[(train_val_negll_c <= train_val_negll_c.quantile(0.25)) & (train_val_negll_I <= train_val_negll_I.quantile(0.25))].sort_values("train_val_negll")
elif objective == "three_objectives":
    train_val_negll_c = (df["train_negll_c"]*df["n_obs_train_c"] + df["val_negll_c"]*df["n_obs_val_c"])/(df["n_obs_train_c"] + df["n_obs_val_c"])
    train_val_negll_I = (df["train_negll_I"]*df["n_obs_train_I"] + df["val_negll_I"]*df["n_obs_val_I"])/(df["n_obs_train_I"] + df["n_obs_val_I"])
    train_val_negll_prev = (df["train_negll_prev"]*df["n_obs_train_prev"] + df["val_negll_prev"]*df["n_obs_val_prev"])/(df["n_obs_train_prev"] + df["n_obs_val_prev"])
    df.loc[:, "train_val_negll"] = train_val_negll_c + train_val_negll_I + train_val_negll_prev
    df_sub = df.loc[(train_val_negll_c <= train_val_negll_c.quantile(0.25)) & (train_val_negll_I <= train_val_negll_I.quantile(0.25)) & (train_val_negll_prev <= train_val_negll_prev.quantile(0.25))].sort_values("train_val_negll")
elif objective == "prev_and_conc":
    train_val_negll_c = (df["train_negll_c"]*df["n_obs_train_c"] + df["val_negll_c"]*df["n_obs_val_c"])/(df["n_obs_train_c"] + df["n_obs_val_c"])
    train_val_negll_prev = (df["train_negll_prev"]*df["n_obs_train_prev"] + df["val_negll_prev"]*df["n_obs_val_prev"])/(df["n_obs_train_prev"] + df["n_obs_val_prev"])
    df.loc[:, "train_val_negll"] = train_val_negll_c + train_val_negll_prev
    df_sub = df.loc[(train_val_negll_c <= train_val_negll_c.quantile(0.25)) & (train_val_negll_prev <= train_val_negll_prev.quantile(0.25))].sort_values("train_val_negll")

df_nsmallest = df_sub.nsmallest(int(len(df) * cutoff_value), 'train_val_negll')
model_ids = df_nsmallest["seed"].tolist()
reporting_delays = df_nsmallest["reporting_delay"].tolist()

# Print ensemble member seeds to a file
with open(f"{out_dir}/ensemble_members.txt", "w") as f:
    print(f"Dropped {len_pre - len_post} of {len_pre} rows due to optimization failure.", file = f)
    print(f"Considering {len(df_nsmallest)} members for ensemble evaluation.", file=f)

pd.DataFrame({"seed": model_ids, "reporting_delay": reporting_delays}).to_csv(
    f"{out_dir}/ensemble_members.csv", index=False
)

base_config = {
        # General settings
        "seed": 0,  # random seed

        # data selection settings
        "data_kwargs": {
            "town": town,
            "log_scale": True, # this only considers WW measurements, not case counts
        },
        
        "dt": 0.2,
        "T_max": 25, # dummy value
        "underreporting_model": "monotone_increasing"
}
with open(hparams_path) as f:
    hparams = json.load(f)

base_config.update(hparams)
base_config["solver_kwargs"] = {
            "rtol": hparams.get("rtol", 1e-4),
            "atol": hparams.get("atol", 1e-6),
        }

base_config["phase_cut_date"] = phase_cut_date
base_config["prev_phase_cut_date"] = prev_phase_cut_date
data = optimization_utils.two_phase_integrative_model_load_data(base_config)

base_config["E0"] = float(data["I_all"][0]/7)*1/0.5
base_config["I0"] = float(data["I_all"][0]/7)*5
base_config["R0"] = 0.92*data["population_size"] # 92% of pop, based on https://www.rki.de/DE/Themen/Infektionskrankheiten/Infektionskrankheiten-A-Z/C/COVID-19-Pandemie/AK-Studien/Ergebnisse.html


data = optimization_utils.two_phase_integrative_model_load_data(base_config)


def get_model_predictions(model, t_all, t_phase_1):
    """
    Collect predictions from the model for all relevant variables.

    Returns a dictionary with keys:
      - "SEIR": array [T, 4]
      - "beta": array [T]
      - "R_eff": array [T]
      - "underreporting_rate": array [T]
      - "I7_reported": array [T-1]
      - "log_concentration": array [T']
      - "test_positive_rate": array [T]
    """
    # ---- SEIR states ----
    seir_states = model.UDE(t_all)[:, :, 0]   # shape (len(t_all), 5) with (S,E,I,R,cum_I_new)
    S, E, I, R, cum_I_new = seir_states.T

    # ---- beta(t) ----
    if len(model.UDE.rhs.freqs) > 0:
        feats = jax.vmap(model.UDE.rhs.fourier_features)(t_all.reshape(-1, 1))
        beta_t = jax.vmap(model.UDE.rhs.mlp)(feats)[:, 0]
    else:
        beta_t = jax.vmap(model.UDE.rhs.mlp)(t_all.reshape(-1, 1))[:, 0]

    # ---- effective reproduction number ----
    gamma = 0.2
    R_eff = beta_t / gamma * S / model.UDE.rhs.N

    # ---- underreporting rate ----
    underrep = jax.vmap(model.underreporting_model)(t_phase_1.reshape(-1, 1))[:, 0]
    underrep = jnp.concatenate([
        underrep,
        jnp.repeat(underrep[-1], len(t_all) - len(t_phase_1))
    ])

    # ---- shedding curve ----
    s = jnp.arange(0, model.T_max+base_config["dt"], base_config["dt"])
    shedding_values = jax.vmap(two_phase_integrative_ude.shedding_curve, in_axes=(0, None, None, None, None))(s, model.logit_T_peak, model.log_k1, model.logit_k2, model.logit_k3)

    # ---- test positive rate ----
    prevalence = (E+I)/model.UDE.rhs.N * 100.0  # Prevalence in percent
    se = 0.83
    sp = 1.00
    pos_rate = se*prevalence + (1.0-sp)*(1-prevalence)


    # ---- wastewater conc. + 7d reported infections ----
    log_conc, I7_rep = model(t_all, t_phase_1)

    return {
        "SEIR": seir_states[:, :4],           # S,E,I,R
        "beta": beta_t,
        "R_eff": R_eff,
        "underreporting_rate": underrep,
        "I7_reported": I7_rep,
        "log_concentration": log_conc,
        "shedding_curve": shedding_values,
        "pos_rate": pos_rate,
    }


def get_ensemble_predictions(model_ids, reporting_delays, t_all, t_phase_1, multistart_result_path, 
                             quantiles=(0.025, 0.5, 0.975)):
    # Collect predictions for each model
    all_SEIR, all_beta, all_R, all_underrep, all_I7, all_logC, all_shedding, all_pos_rate, all_noise_params = [], [], [], [], [], [], [], [], []
    for model_id, reporting_delay in zip(model_ids, reporting_delays):
        base_model = two_phase_integrative_ude.IntegrativeModel(
            width_size=base_config["width_size"],
            depth=base_config["depth"],
            activation=model_utils.activation_fct_mapper[base_config["activation"]],
            n_freqs=base_config["n_freqs"],
            t_scale=data["t_scale"],
            population_size=data["population_size"],
            E0_init=base_config["E0"],
            I0_init=base_config["I0"],
            R0_init=base_config["R0"],
            solver=model_utils.solver_mapper[base_config["solver"]],
            solver_kwargs=base_config["solver_kwargs"],
            k1_init=base_config["k1_init"],
            k2_init=base_config["k2_init"],
            k3_init=base_config["k3_init"],
            T_peak_init=base_config["T_peak_init"],
            T_max=base_config["T_max"],
            dt = base_config["dt"],
            key=jr.key(model_id),
            init_par_vmr=base_config.get("init_par_vmr", 1.0),
            init_sigma_C=base_config.get("init_sigma_C", 1.0),
            reporting_delay=reporting_delay,
            underreporting_model=base_config.get("underreporting_model", "none")) # days, delay between infection and reporting

        m = eqx.tree_deserialise_leaves(f"{multistart_result_path}/{model_id}_model.eqx", base_model)
        preds = get_model_predictions(m, t_all, t_phase_1)
        all_SEIR.append(preds["SEIR"])
        all_beta.append(preds["beta"])
        all_R.append(preds["R_eff"])
        all_underrep.append(preds["underreporting_rate"])
        all_I7.append(preds["I7_reported"])
        all_logC.append(preds["log_concentration"])
        all_shedding.append(preds["shedding_curve"])
        all_pos_rate.append(preds["pos_rate"])
        all_noise_params.append(jnp.asarray((
            1.0 + jax.nn.softplus(m.par_vmr),
            jnp.exp(m.log_sigma_C),
        )))

    # Stack arrays along ensemble axis
    all_SEIR = jnp.stack(all_SEIR)            # [n_models, T, 4]
    all_beta = jnp.stack(all_beta)            # [n_models, T]
    all_R = jnp.stack(all_R)                  # [n_models, T]
    all_underrep = jnp.stack(all_underrep)    # [n_models, T]
    all_I7 = jnp.stack(all_I7)                # [n_models, T-1]
    all_logC = jnp.stack(all_logC)            # [n_models, T']
    all_shedding = jnp.stack(all_shedding)    # [n_models, T_max+1]
    all_pos_rate = jnp.stack(all_pos_rate)    # [n_models, T]
    all_noise_params = jnp.stack(all_noise_params)  # [n_models, n_noise_parameters]

    def summary(arr):
        return {
            "all": arr,
            "mean": jnp.mean(arr, axis=0),
            "quantiles": {q: jnp.quantile(arr, q, axis=0) for q in quantiles}
        }

    return {
        "SEIR": summary(all_SEIR),
        "beta": summary(all_beta),
        "R_eff": summary(all_R),
        "underreporting_rate": summary(all_underrep),
        "I7_reported": summary(all_I7),
        "log_concentration": summary(all_logC),
        "shedding_curve": summary(all_shedding),
        "test_positive_rate": summary(all_pos_rate),
        "noise_parameters": all_noise_params,
    }


def plot_model_ensemble(ensemble_preds,
                        phase_cut_date,
                        I_dates_train, I_train,
                        I_dates_val, I_val,
                        obs_dates_phase_2, obs_I_phase_2, dates_all,
                        conc_dates_train, conc_train,
                        conc_dates_val, conc_val,
                        t_all_idx):

    # --- Reported cases ---
    I7_median = ensemble_preds["I7_reported"]["quantiles"][0.5]
    I7_low  = ensemble_preds["I7_reported"]["quantiles"][0.025]
    I7_high = ensemble_preds["I7_reported"]["quantiles"][0.975]

    fig, ax = plt.subplots(figsize=(9, 3), dpi=300)
    ax.scatter(I_dates_train, I_train, label="Training", color="goldenrod", s=10)
    ax.scatter(I_dates_val, I_val, label="Validation", color="black", s=10)
    ax.scatter(obs_dates_phase_2, obs_I_phase_2, label="Test", color="#595959", s=10, alpha=0.8)
    ax.axvline(pd.to_datetime(phase_cut_date), color="#595959", linestyle='--', label="Phase split")

    ax.fill_between(dates_all[1:], I7_low, I7_high, color="#8B0000", alpha=0.15, label="95% CI")
    I7_low  = ensemble_preds["I7_reported"]["quantiles"][0.05]
    I7_high = ensemble_preds["I7_reported"]["quantiles"][0.95]
    ax.fill_between(dates_all[1:], I7_low, I7_high, color="#8B0000", alpha=0.3, label="90% CI")
    I7_low  = ensemble_preds["I7_reported"]["quantiles"][0.25]
    I7_high = ensemble_preds["I7_reported"]["quantiles"][0.75]
    ax.fill_between(dates_all[1:], I7_low, I7_high, color="#8B0000", alpha=0.45, label="50% CI")
    ax.plot(dates_all[1:], I7_median, label="Median", color="#8B1000")

    # legend to the right
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 5, 9)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
    plt.ylabel("7-day moving sum of\nnew infections [#]")
    fig.tight_layout(rect=[0, 0, 0.85, 1])

    # --- Wastewater concentrations ---
    conc_median = ensemble_preds["log_concentration"]["quantiles"][0.5]
    conc_low  = ensemble_preds["log_concentration"]["quantiles"][0.025]
    conc_high = ensemble_preds["log_concentration"]["quantiles"][0.975]

    fig2, ax2 = plt.subplots(figsize=(9, 3), dpi=300)
    ax2.scatter(conc_dates_train, conc_train, label="Training", color="royalblue", s=10)
    ax2.scatter(conc_dates_val, conc_val, label="Validation", color="black", s=10)
    ax2.axvline(pd.to_datetime(phase_cut_date), color="#595959", linestyle='--', label="Phase split")

    valid_idx = (t_all_idx*base_config["dt"] >= int(hparams.get("T_max")))
    ax2.fill_between(dates_all[valid_idx], conc_low, conc_high, color="#8B0000", alpha=0.15, label="95% CI")
    conc_low  = ensemble_preds["log_concentration"]["quantiles"][0.05]
    conc_high = ensemble_preds["log_concentration"]["quantiles"][0.95]
    ax2.fill_between(dates_all[valid_idx], conc_low, conc_high, color="#8B0000", alpha=0.3, label="90% CI")
    conc_low  = ensemble_preds["log_concentration"]["quantiles"][0.25]
    conc_high = ensemble_preds["log_concentration"]["quantiles"][0.75]
    ax2.fill_between(dates_all[valid_idx], conc_low, conc_high, color="#8B0000", alpha=0.45, label="50% CI")
    ax2.plot(dates_all[valid_idx], conc_median, label="Median", color="#8B1000")

    ax2.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 5, 9)))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
    plt.ylabel("Flow normalized\nconcentration [copies/l]")
    fig2.tight_layout(rect=[0, 0, 0.85, 1])

    return fig, fig2


def plot_SEIR_ensemble(ensemble_preds, dates_all, phase_cut_date):
    SEIR_median = ensemble_preds["SEIR"]["quantiles"][0.5]
    SEIR_low  = ensemble_preds["SEIR"]["quantiles"][0.025]
    SEIR_high = ensemble_preds["SEIR"]["quantiles"][0.975]

    fig, axs = plt.subplots(ncols=4, figsize=(8.5, 2.5), dpi=300)
    labels = [r"$S$", r"$E$", r"$I$", r"$R$"]
    colors = ["saddlebrown", "peru", "darkgoldenrod", "goldenrod"]

    for i in range(4):
        axs[i].axvline(pd.to_datetime(phase_cut_date), color="#595959", linestyle='--', label="Phase split")
        axs[i].fill_between(dates_all[7:], SEIR_low[7:, i], SEIR_high[7:, i], color=colors[i], alpha=0.15, label="95% CI")
        SEIR_low  = ensemble_preds["SEIR"]["quantiles"][0.05]
        SEIR_high = ensemble_preds["SEIR"]["quantiles"][0.95]
        axs[i].fill_between(dates_all[7:], SEIR_low[7:, i], SEIR_high[7:, i], color=colors[i], alpha=0.3, label="90% CI")
        SEIR_low  = ensemble_preds["SEIR"]["quantiles"][0.25]
        SEIR_high = ensemble_preds["SEIR"]["quantiles"][0.75]
        axs[i].fill_between(dates_all[7:], SEIR_low[7:, i], SEIR_high[7:, i], color=colors[i], alpha=0.45, label="50% CI")
        axs[i].plot(dates_all[7:], SEIR_median[7:, i], color=colors[i], label="Median")
        axs[i].set_title(labels[i])
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
        axs[i].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 5, 9)))
        axs[i].tick_params(axis='x', rotation=45)

        formatter = mticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 3))
        axs[i].yaxis.set_major_formatter(formatter)
        if i == 0:
            axs[i].set_ylabel("Compartment size [#]")

    # figure-level legend on the right
    handles, labs = axs[-1].get_legend_handles_labels()
    fig.legend(handles, labs, loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 1])
    return fig


def plot_beta_ensemble(ensemble_preds, dates_all, phase_cut_date):
    beta_median = ensemble_preds["beta"]["quantiles"][0.5]
    beta_low  = ensemble_preds["beta"]["quantiles"][0.025]
    beta_high = ensemble_preds["beta"]["quantiles"][0.975]

    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=300)
    ax.axvline(pd.to_datetime(phase_cut_date), color="#595959", linestyle='--', label="Phase split")
    ax.fill_between(dates_all[10:], beta_low[10:], beta_high[10:], color="saddlebrown", alpha=0.15, label="95% CI")
    beta_low  = ensemble_preds["beta"]["quantiles"][0.05]
    beta_high = ensemble_preds["beta"]["quantiles"][0.95]
    ax.fill_between(dates_all[10:], beta_low[10:], beta_high[10:], color="saddlebrown", alpha=0.3, label="90% CI")
    beta_low  = ensemble_preds["beta"]["quantiles"][0.25]
    beta_high = ensemble_preds["beta"]["quantiles"][0.75]
    ax.fill_between(dates_all[10:], beta_low[10:], beta_high[10:], color="saddlebrown", alpha=0.45, label="50% CI")
    ax.plot(dates_all[10:], beta_median[10:], c="saddlebrown", label="Median")

    ax.set_ylabel(r"$\beta$")
    ax.tick_params(axis='x', rotation=45)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 5, 9)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))

    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout(rect=[0, 0, 0.86, 1])
    return fig


def plot_Rt_ensemble(ensemble_preds, dates_all, phase_cut_date):
    Rt_median = ensemble_preds["R_eff"]["quantiles"][0.5]
    Rt_low  = ensemble_preds["R_eff"]["quantiles"][0.025]
    Rt_high = ensemble_preds["R_eff"]["quantiles"][0.975]

    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=300)
    ax.axvline(pd.to_datetime(phase_cut_date), color="#595959", linestyle='--', label="Phase split")
    ax.fill_between(dates_all[7:], Rt_low[7:], Rt_high[7:], color="saddlebrown", alpha=0.15, label="95% CI")
    Rt_low  = ensemble_preds["R_eff"]["quantiles"][0.05]
    Rt_high = ensemble_preds["R_eff"]["quantiles"][0.95]
    ax.fill_between(dates_all[7:], Rt_low[7:], Rt_high[7:], color="saddlebrown", alpha=0.3, label="90% CI")
    Rt_low  = ensemble_preds["R_eff"]["quantiles"][0.25]
    Rt_high = ensemble_preds["R_eff"]["quantiles"][0.75]
    ax.fill_between(dates_all[7:], Rt_low[7:], Rt_high[7:], color="saddlebrown", alpha=0.45, label="50% CI")
    ax.axhline(1.0, color="#A1A1A1", linestyle='--')
    ax.plot(dates_all[7:], Rt_median[7:], c="saddlebrown", label="Median")
    ax.set_ylabel(r"$R_t$")
    ax.tick_params(axis='x', rotation=45)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 5, 9)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))

    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout(rect=[0, 0, 0.86, 1])
    return fig


def plot_underreporting_ensemble(ensemble_preds, dates_all, phase_cut_date):
    underrep_median = ensemble_preds["underreporting_rate"]["quantiles"][0.5]
    underrep_low  = ensemble_preds["underreporting_rate"]["quantiles"][0.025]
    underrep_high = ensemble_preds["underreporting_rate"]["quantiles"][0.975]

    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=300)
    ax.axvline(pd.to_datetime(phase_cut_date), color="#595959", linestyle='--', label="Phase split")
    ax.fill_between(dates_all[7:], (1-underrep_high[7:])*100, (1-underrep_low[7:])*100, color="grey", alpha=0.15, label="95% CI")
    underrep_low  = ensemble_preds["underreporting_rate"]["quantiles"][0.05]
    underrep_high = ensemble_preds["underreporting_rate"]["quantiles"][0.95]
    ax.fill_between(dates_all[7:], (1-underrep_high[7:])*100, (1-underrep_low[7:])*100, color="grey", alpha=0.3, label="90% CI")
    underrep_low  = ensemble_preds["underreporting_rate"]["quantiles"][0.25]
    underrep_high = ensemble_preds["underreporting_rate"]["quantiles"][0.75]
    ax.fill_between(dates_all[7:], (1-underrep_high[7:])*100, (1-underrep_low[7:])*100, color="grey", alpha=0.45, label="50% CI")
    ax.plot(dates_all[7:], (1-underrep_median[7:])*100, c="black", label="Median")

    ax.set_ylabel("Reporting rate [%]")
    ax.set_ylim(0, 100)
    ax.tick_params(axis='x', rotation=45)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 5, 9)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))

    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout(rect=[0, 0, 0.86, 1])
    return fig


def plot_shedding_curve_ensemble(ensemble_preds, T_max, dt):
    shedding_median = ensemble_preds["shedding_curve"]["quantiles"][0.5]
    shedding_low  = ensemble_preds["shedding_curve"]["quantiles"][0.025]
    shedding_high = ensemble_preds["shedding_curve"]["quantiles"][0.975]

    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=300)
    s = jnp.arange(0, T_max+dt, dt)
    ax.fill_between(s, shedding_low, shedding_high, color="goldenrod", alpha=0.15, label="95% CI")
    shedding_low  = ensemble_preds["shedding_curve"]["quantiles"][0.05]
    shedding_high = ensemble_preds["shedding_curve"]["quantiles"][0.95]
    ax.fill_between(s, shedding_low, shedding_high, color="goldenrod", alpha=0.3, label="90% CI")
    shedding_low  = ensemble_preds["shedding_curve"]["quantiles"][0.25]
    shedding_high = ensemble_preds["shedding_curve"]["quantiles"][0.75]
    ax.fill_between(s, shedding_low, shedding_high, color="goldenrod", alpha=0.45, label="50% CI")
    ax.plot(s, shedding_median, c="goldenrod", label="Median")
    ax.set_xlabel("Days since becoming infectious")
    ax.set_ylabel("Shedding rate intensity")

    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout(rect=[0, 0, 0.86, 1])
    return fig


def plot_test_positive_rate_ensemble(ensemble_preds, dates_all, prev_phase_cut_date, data):
    pos_rate_median = ensemble_preds["test_positive_rate"]["quantiles"][0.5]


    fig, ax = plt.subplots(figsize=(6, 2.5), dpi=300)
    ax.scatter(data["prevalence_dates_train"], data["pos_tests_train"]/data["n_tests_train"]*100, color="#63A066", s=10, label="Training")
    ax.scatter(data["prevalence_dates_val"], data["pos_tests_val"]/data["n_tests_val"]*100, color="#043507", s=10, label="Validation")
    ax.scatter(data["prevalence_dates_test"], data["pos_tests_test"]/data["n_tests_test"]*100, color="#595959", s=10, label="Test")
    ax.axvline(pd.to_datetime(prev_phase_cut_date), color="#595959", linestyle='--', label="Phase split")

    ax.plot(dates_all, pos_rate_median, c="saddlebrown", label="Median")
    
    pos_rate_low  = ensemble_preds["test_positive_rate"]["quantiles"][0.25]
    pos_rate_high = ensemble_preds["test_positive_rate"]["quantiles"][0.75]
    ax.fill_between(dates_all, pos_rate_low, pos_rate_high, color="saddlebrown", alpha=0.45, label="50% CI")

    pos_rate_low  = ensemble_preds["test_positive_rate"]["quantiles"][0.05]
    pos_rate_high = ensemble_preds["test_positive_rate"]["quantiles"][0.95]
    ax.fill_between(dates_all, pos_rate_low, pos_rate_high, color="saddlebrown", alpha=0.3, label="90% CI")

    pos_rate_low  = ensemble_preds["test_positive_rate"]["quantiles"][0.025]
    pos_rate_high = ensemble_preds["test_positive_rate"]["quantiles"][0.975]
    ax.fill_between(dates_all, pos_rate_low, pos_rate_high, color="saddlebrown", alpha=0.15, label="95% CI")


    ax.set_ylabel("Test positivity [%]")
    ax.tick_params(axis='x', rotation=45)
    # ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 5, 9)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))

    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.tight_layout(rect=[0, 0, 0.86, 1])
    return fig


ensemble_predictions = get_ensemble_predictions(model_ids, reporting_delays, data["t_all"], data["t_phase_1"],
                                                multistart_path, quantiles=(0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975)) # 50%, 90%, 95% confidence intervals 

for key, value in ensemble_predictions.items():
    if isinstance(value, dict):
        jnp.savez(f"{out_dir}/ensemble_predictions_{key}.npz", **value)
    else:
        jnp.savez(f"{out_dir}/ensemble_predictions_{key}.npz", all=value)



fig1, fig2 = plot_model_ensemble(ensemble_predictions,
                        phase_cut_date,
                        data["I_dates_train"], data["I_train"],
                        data["I_dates_val"], data["I_val"],
                        data["obs_dates_phase_2"], data["I_test"], data["dates_all"],
                        data["conc_dates_train"], data["conc_train"],
                        data["conc_dates_val"], data["conc_val"],
                        data["t_all_idx"])

fig3 = plot_SEIR_ensemble(ensemble_predictions, data["dates_all"], phase_cut_date)
fig4 = plot_beta_ensemble(ensemble_predictions, data["dates_all"], phase_cut_date)
fig5 = plot_Rt_ensemble(ensemble_predictions, data["dates_all"], phase_cut_date)
fig6 = plot_underreporting_ensemble(ensemble_predictions, data["dates_all"], phase_cut_date)
fig7 = plot_shedding_curve_ensemble(ensemble_predictions, base_config["T_max"], base_config["dt"])
fig8 = plot_test_positive_rate_ensemble(ensemble_predictions, data["dates_all"], prev_phase_cut_date, data)

# ensure legends outside the axes are included
fig1.savefig(f"{out_dir}/multistart_ensemble_1_I.png", bbox_inches='tight')
fig2.savefig(f"{out_dir}/multistart_ensemble_2_conc.png", bbox_inches='tight')
fig3.savefig(f"{out_dir}/multistart_ensemble_3_SEIR.png", bbox_inches='tight')
fig4.savefig(f"{out_dir}/multistart_ensemble_4_beta.png", bbox_inches='tight')
fig5.savefig(f"{out_dir}/multistart_ensemble_5_Rt.png", bbox_inches='tight')
fig6.savefig(f"{out_dir}/multistart_ensemble_6_underreporting.png", bbox_inches='tight')
fig7.savefig(f"{out_dir}/multistart_ensemble_7_shedding.png", bbox_inches='tight')
fig8.savefig(f"{out_dir}/multistart_ensemble_8_test_positivity.png", bbox_inches='tight')
