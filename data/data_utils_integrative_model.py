import pandas as pd
import jax.numpy as jnp
from .data_utils import load_data as load_ww_data
import os
import warnings
# add this import
from pathlib import Path


def load_data(data_kwargs, t_normalized=True):
    town = data_kwargs["town"]
    
    # some form of normalization helps the NN
    min_train_date = pd.Timestamp("2022-02-28")
    max_train_date = pd.Timestamp("2023-03-29")
    t_scale = (max_train_date - min_train_date).days
    
    if town != "Bonn":
        # build path relative to this file
        DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "preprocessed"
        df = pd.read_csv(DATA_DIR / "rhineland_palatinate_data.csv")
        df = df.loc[df.Town == town].drop(columns=["Town"])
        df["Date"] = pd.to_datetime(df["Date"])
        df["day_idx"] = (df.Date-df.Date.min()).dt.days
        assert df.loc[df["Faelle_7-Tage"].isna()].__len__()==0, "There are NaN values in the case counts data, the data pipeline has to be adapted"
        prevalence = df.loc[df.prevalence.notna(), ["day_idx", "Valid test", "Positive test", "prevalence", "Date"]]
        # "If there is no measurement available at that day, the time series is filled with 0." (https://www.nature.com/articles/s41598-024-64864-1#Sec16)
        df = df.loc[df["Anteil Genkopien Durchschnitt zu PMMoV x100.000"]>0]    
        population_size = int(df.Bevoelkerung.unique()[0])
        if t_normalized:
            assert "Data pipeline is not implemented with a pre-normalized t for SentiSurv data"
        if not data_kwargs["log_scale"]:
            assert "Data pipeline is not implemented without log scale for SentiSurv data"

        df["log_concentration"] = jnp.log(df["Anteil Genkopien Durchschnitt zu PMMoV x100.000"].values)
        return None, df.Date.values, t_scale, df["day_idx"].values, df[["log_concentration", "Faelle_7-Tage"]].values, population_size, prevalence
    else:
        if data_kwargs.get("pathogen", "COV19") == "influenza":
            # build path relative to this file
            DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "preprocessed"
            df = pd.read_csv(DATA_DIR / "influenza_Bonn.csv")
            df["Date"] = pd.to_datetime(df["Date"])
            df["day_idx"] = (df.Date-df.Date.min()).dt.days

            population_size = df["population"].mean()

           #  df["log_concentration"] = jnp.log(df["Avg_flow_normalized_Flu"].values)
            return None, df.Date.values, t_scale, df["day_idx"].values, df[["flow_normalized_Flu_combined", "cases"]].values, population_size, None

        # load wastwater data
        conc_y_name, date, total_train_seconds, ts, ys = load_ww_data(**data_kwargs, return_raw_t=True)
        df_conc = pd.DataFrame({"Date": date, conc_y_name: ys[:,0], "ts": ts})#

        # load case counts data
        here = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(here, "preprocessed", "case_counts.csv")
        case_counts = pd.read_csv(csv_path)
        case_counts["Meldedatum"] = pd.to_datetime(case_counts["Meldedatum"], format="%Y-%m-%d")
        case_counts = case_counts.groupby("Meldedatum")[["N", "N_7d"]].sum().reset_index()#

        # merge
        df_res = df_conc.merge(case_counts, left_on="Date", right_on="Meldedatum", how="outer", suffixes=("", "_case_counts"))
        # unify date information
        df_res["Date"] = df_res.apply(lambda x: x["Date"] if pd.notna(x["Date"]) else x["Meldedatum"], axis=1)
        df_res.drop(columns=["Meldedatum"], inplace=True)
        # linearly interpolate missing t normalized values according to date information
        df_res["ts"] = df_res.set_index("Date").ts.interpolate(method="time").reset_index(drop=True)

        df_res["day_idx"] = (df_res.Date-df_res.Date.min()).dt.days


        if data_kwargs["sampling_area"]=="North":
            population_size = 87573
        elif data_kwargs["sampling_area"]=="South":
            population_size = 88614
        elif data_kwargs["sampling_area"]=="North_South":
            population_size = 88614 + 87573
        
        if (df_res[["N_7d"]].values==0).sum()!=0:
            warnings.warn("There are zero values in the ys[:,1] array. This may lead to issues with the grad_loss function, as the number of observations is calculated based on the # observations!=0 for easy jit.")

        if t_normalized:
            return conc_y_name, df_res["Date"].values, t_scale, df_res["ts"].values, df_res[[conc_y_name, "N_7d"]].values, population_size, None
        else:
            return conc_y_name, df_res["Date"].values, t_scale, df_res["day_idx"].values, df_res[[conc_y_name, "N_7d"]].values, population_size, None