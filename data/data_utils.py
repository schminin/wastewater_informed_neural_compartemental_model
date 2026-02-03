import pandas as pd
import jax
import jax.numpy as jnp
import jax.random as jr
import os

def load_data(sampling_area, 
              project, # both, ESI_CorA or AMELAG
              max_precipitation_subsetting, substance_normalization, 
              gene_target, log_scale, return_raw_t=False, **kwargs):
    # Define fixed training time range
    min_train_date = pd.Timestamp("2022-02-28")
    max_train_date = pd.Timestamp("2023-03-29")
    total_train_seconds = (max_train_date - min_train_date).total_seconds()
    
    # Load data
    file_dir = os.path.dirname(__file__)  # path to this script file
    data_path = os.path.join(file_dir,"preprocessed")
    df = pd.read_csv(f'{data_path}/wastewater.csv')
    df = df.iloc[1:,:]
    df["Date"] = pd.to_datetime(df["Date"])  # ensure datetime dtype

    # Select appropriate substance column
    substance_column = f"COVID_{gene_target}"
    if substance_normalization:
        substance_column = f"{substance_normalization}_normalized_{substance_column}"

    # Subselect by sampling area
    df_sub = df.loc[df.Sampling_Area == sampling_area].copy()

    # Subselect by project
    if project != "both":
        df_sub = df_sub.loc[df_sub.Project == project]
    elif sampling_area == "North_South":
        # for some timepoints of North_South, we have observations for both projects
        # in these cases, we only want to keep the ESI_CorA project
        df_sel = df_sub.groupby(["Date", "Sampling_Area"]).count().reset_index()
        sel_dates = df_sel.loc[df_sel[substance_column]>1, "Date"]
        df_sub = df_sub.loc[~(df_sub.Date.isin(sel_dates) & (df_sub.Project=="AMELAG")), :]
        if len(df_sub)!=len(df_sub.Date.unique()):
            raise ValueError("there are date duplicates")

    # Subselect by precipitation setting
    if max_precipitation_subsetting == "dry":
        df_sub = df_sub.loc[df_sub.precipitation_event == "dry"]
    elif max_precipitation_subsetting == "light_rain":
        df_sub = df_sub.loc[df_sub.precipitation_event != "heavy_rain"]

    # Normalize time
    df_sub["t_normalized"] = (
        (df_sub["Date"] - min_train_date).dt.total_seconds() / total_train_seconds
    )
    df_sub.reset_index(drop=True, inplace=True)
    df_sub = df_sub[["Date", substance_column, "t_normalized"]]
    
    #print(f"Min date: {df_sub.Date.min()}")
    #print(f"Max date: {df_sub.Date.max()}")
    if log_scale:
        if return_raw_t:
            return "log_"+substance_column, df_sub["Date"].values, total_train_seconds, df_sub["t_normalized"].values, jnp.log(df_sub[substance_column].values).reshape(-1, 1)
        else:
            return "log_"+substance_column, df_sub["t_normalized"].values, jnp.log(df_sub[substance_column].values).reshape(-1, 1)
    else:
        if return_raw_t:
            return substance_column, df_sub["Date"].values, df_sub["t_normalized"].values, df_sub[substance_column].values.reshape(-1, 1)
        else:
            return substance_column, df_sub["t_normalized"].values, df_sub[substance_column].values.reshape(-1, 1)



def split_dataset(ts: jax.Array, ys: jax.Array, key: jax.Array, train_frac: float = 0.9):
    n_timepoints = ts.shape[0]

    # Always include t=0 in training set
    perm = jr.permutation(key, n_timepoints - 1)
    split_idx = int(train_frac * n_timepoints)

    # Shift indices by 1 (to skip t=0), then add 0 back to training set
    train_perm = jnp.sort(jnp.concatenate([jnp.array([0]), perm[:split_idx] + 1]))
    test_perm = jnp.sort(perm[split_idx:] + 1)

    train_ts = ts[train_perm]
    train_obs = ys[train_perm]
    test_ts = ts[test_perm]
    test_obs = ys[test_perm]

    return {
        "y0_baseline": jnp.mean(ys[0:10]),  # initial condition
        "train": (train_perm, train_ts, train_obs),
        "test": (test_perm, test_ts, test_obs),
    }


def dataloader_1d(
    train_ts: jax.Array,
    train_obs: jax.Array,
    key: jax.Array,
    frac: float = 0.8,
    always_include_first: bool = False,
):
    """
    Infinite generator that returns random subsets of time series.
    
    Args:
        train_ts: 1D array of time points
        train_obs: 1D array of corresponding observations (same length)
        key: PRNG key
        frac: fraction of timepoints to select per batch
        always_include_first: if True, always include index 0
    
    Yields:
        (batch_ts, batch_obs) tuples
    """
    n = train_ts.shape[0]

    while True:
        key, subkey = jr.split(key)
        
        # Exclude index 0 if needed
        valid_indices = jnp.arange(1, n) if always_include_first else jnp.arange(n)
        num_samples = int(frac * n) - int(always_include_first)

        # Sample random indices
        sampled = jr.choice(subkey, valid_indices, shape=(num_samples,), replace=False)

        if always_include_first:
            idx = jnp.sort(jnp.concatenate([jnp.array([0]), sampled]))
        else:
            idx = jnp.sort(sampled)

        yield train_ts[idx], train_obs[idx]
