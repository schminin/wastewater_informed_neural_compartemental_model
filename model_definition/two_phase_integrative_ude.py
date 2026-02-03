# Define the UDE function
import jax
import jax.nn as jnn
import equinox as eqx
import diffrax
import jax.numpy as jnp

import pandas as pd
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from typing import Callable, Sequence

class UDE_rhs(eqx.Module):
    mlp: eqx.nn.MLP        # Neural network modeling unknown dynamics
    freqs: tuple[float,...] = eqx.field(static=True)
    t_scale: float = eqx.field(static=True)
    N: int = eqx.field(static=True)


    def __init__(self, state_dim, width_size, depth, activation, 
                 n_freqs, t_scale, population_size, *, key, **kwargs):
        super().__init__(**kwargs)
        freq_key = jax.random.key(123)
        init_freqs = jnp.linspace(7.0, 12.0, n_freqs) + 0.5 * jax.random.normal(freq_key, (n_freqs,))
        self.freqs = tuple(init_freqs.tolist())
        self.mlp = eqx.nn.MLP(
            in_size=1+n_freqs*2, # t as input and 2*n_freqs Fourier features
            out_size=state_dim, # state_dim as output
            width_size=width_size,
            depth=depth,
            activation=activation,  # Smooth activation for continuous dynamics
            final_activation=jnp.exp,
            key=key,
        )
        self.t_scale = t_scale
        self.N = population_size

    def fourier_features(self, t):
        freqs_arr = jnp.array(self.freqs)
        t = jnp.atleast_1d(t)
        cosinus = jnp.cos(t * freqs_arr)
        sinus = jnp.sin(t * freqs_arr)
        return jnp.concatenate([cosinus, sinus, t], axis=-1)
    
    def beta(self, t):
        if len(self.freqs)>0:
            freqs_arr = jnp.array(self.freqs)
            t_feat = jnp.matmul(jnp.expand_dims(t, -1), jnp.expand_dims(freqs_arr, -1).T)
            mlp_input = jnp.concatenate([jnp.cos(t_feat), jnp.sin(t_feat), jnp.expand_dims(t, -1)], axis=-1)
        else:
            mlp_input = jnp.expand_dims(t,-1)
        return jax.vmap(self.mlp)(mlp_input)[:,0] * self.t_scale # return beta


    def __call__(self, t, y, args):
        
        # possibly add Fourier features to time input
        if len(self.freqs)>0:
            t_feat = self.fourier_features(t) 
        else:
            t_feat = jnp.atleast_1d(t)

        S, E, I, R, cum_I_new = y
        beta = self.mlp(t_feat)*self.t_scale  # Infection rate
        
        # as parameter are normally reported in unit 1/days and our t is on a different scale we have to rescale these
        # good source to adapt parameter values accordingly: https://assets.publishing.service.gov.uk/media/641c7a9b32a8e0000cfa9327/COVID-19-infectiousness-_asymptomatic-transmission.pdf
        mu = 0.007*self.t_scale # waning immunity rate, 120-180 days until vaccine or infection-induced protection against infection wanes
        alpha = 0.5*self.t_scale # 1.7 – 2.5 days latent incubation period, 
        gamma = 0.2*self.t_scale # around 5 days "Ausscheidung vermehrungsfähige Viren" https://www.rki.de/DE/Aktuelles/Publikationen/RKI-Ratgeber/Ratgeber/Ratgeber_COVID-19.html?nn=16777040#doc16925338bodyText8

        dS = -beta * S * I/self.N + mu  * R
        dE = beta * S * I/self.N - alpha * E
        dI = alpha * E - gamma * I
        dR = gamma * I - mu * R     
        dI_new = alpha * E  # New infections from exposed individuals
  
        return jnp.array([dS, dE, dI, dR, dI_new])  # Return the derivatives as a jax array


# Neural ODE wrapper using Diffrax solver
class NeuralUDE(eqx.Module):
    rhs: UDE_rhs
    solver: diffrax.AbstractSolver
    rtol: float = eqx.field(static=True)
    atol: float = eqx.field(static=True)
    E0:  jax.Array
    I0:  jax.Array
    R0:  jax.Array
    population_size: int = eqx.field(static=True)  # Population size for scaling parameters

    def __init__(self, state_dim, width_size, depth, activation, n_freqs, t_scale, population_size, E0_init, I0_init, R0_init,
                 solver, solver_kwargs, *, key, **kwargs):
        super().__init__(**kwargs)
        self.rhs = UDE_rhs(state_dim, width_size, depth, activation, n_freqs, t_scale, population_size, key=key)
        self.solver = solver
        self.rtol = solver_kwargs["rtol"]
        self.atol = solver_kwargs["atol"]
        self.E0 = jnp.array([E0_init])
        self.I0 = jnp.array([I0_init])
        self.R0 = jnp.array([R0_init])
        self.population_size = population_size

    def get_y0(self):
        alpha = 0.5
        return jnp.concat([jnp.array([self.population_size])-self.E0-self.I0-self.R0, self.E0, self.I0, self.R0, jnp.array([0.0])]).reshape(-1,1)  # Initial conditions: S, E, I, R, cum_I_new
           
    def __call__(self, ts):
        # Use learnable self.y0 instead of external input
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.rhs),
            self.solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=None,
            y0=self.get_y0(),  # Initial conditions: S=1, E=0, I=E0, R=I0, cum_I_new=0,                                      
            stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
            saveat=diffrax.SaveAt(ts=ts),
            max_steps=10000,
            throw= False,  # Do not throw an error on failure, just return NaNs
        )
        return solution.ys


def _vs_init(key, shape):
    # variance scaling init for raw params (works well with softplus reparam)
    if len(shape) == 1:
        # reasonable 1-D init (fan_avg ~ shape[0])
        return jax.random.normal(key, shape) * jnp.sqrt(1.0 / shape[0])
    return jax.nn.initializers.variance_scaling(scale=1.0, mode="fan_avg", distribution="truncated_normal")(key, shape)

class MonotoneIncreasingMLP(eqx.Module):
    Ws_raw: list
    bs: list
    w_out_raw: jnp.ndarray
    b_out: jnp.ndarray
    activation: Callable
    final_activation: Callable | None

    def __init__(
        self,
        in_size: int = 1,
        width_size: int = 64,
        depth: int = 2,                  # number of hidden layers
        activation: Callable = jax.nn.tanh,
        final_activation: Callable | None = jax.nn.sigmoid,
        key = jax.random.PRNGKey(0),
    ):
        assert in_size == 1, "This module assumes a 1D input t."
        assert depth >= 1, "Use at least one hidden layer."
        assert activation not in ["tanh", "sigmoid", "relu"], "Activation function must be monotone increasing."
        assert final_activation not in ["tanh", "sigmoid", "relu"], "Final activation function must be monotone increasing."
        
        self.activation = activation
        self.final_activation = final_activation

        keys = jax.random.split(key, depth + 2)  # hidden Ws, hidden bs, final w, final b
        # Hidden layers: sizes 1 -> width -> ... -> width
        sizes: Sequence[int] = [in_size] + [width_size] * depth

        Ws_raw = []
        bs = []
        for i in range(depth):
            kW, kb = jax.random.split(keys[i])
            Wshape = (sizes[i+1], sizes[i])     # (out, in)
            Bshape = (sizes[i+1],)
            Ws_raw.append(_vs_init(kW, Wshape))
            bs.append(jnp.zeros(Bshape))

        self.w_out_raw = _vs_init(keys[depth], (sizes[-1],))
        self.b_out = jnp.array(0.0)

        self.Ws_raw = Ws_raw
        self.bs = bs

    def __call__(self, t: jnp.ndarray) -> jnp.ndarray:
        """t shape: (...,) or (..., 1)"""
        t = jnp.atleast_1d(t)
        if t.ndim == 1:
            t = t[:, None]  # (...,1)

        # Forward through hidden layers with non-negative weights
        h = t
        for W_raw, b in zip(self.Ws_raw, self.bs):
            W = jax.nn.softplus(W_raw)          # ensure elementwise >= 0
            h = h @ W.T + b                     # affine
            h = self.activation(h)              # monotone increasing activation

        # Final linear with non-positive weights -> overall increasing w.r.t t
        w_out = jax.nn.softplus(self.w_out_raw)   # elementwise <= 0, shape (width,)
        z = (h * w_out).sum(axis=-1) + self.b_out  # shape (...,)

        return self.final_activation(z) if self.final_activation is not None else z


class IntegrativeModel(eqx.Module):
    UDE: NeuralUDE
    log_k1: jax.Array
    logit_k2: jax.Array
    logit_k3: jax.Array
    logit_T_peak: jax.Array # float = eqx.field(static=True)
    T_max: float = eqx.field(static=True)
    t_scale: float = eqx.field(static=True)
    dt: float = eqx.field(static=True)
    underreporting_act: str = eqx.field(static=True)
    underreporting_width: int = eqx.field(static=True)
    underreporting_depth: int = eqx.field(static=True)
    underreporting_model: eqx.Module
    reporting_delay: int = eqx.field(static=True)  # days, delay between infection and reporting
    par_vmr: jax.Array                  # trainable log‐noise for cases
    log_sigma_C: jax.Array                  # trainable log‐noise for concentration

    def __init__(self, width_size, depth, activation, n_freqs, t_scale, population_size, E0_init, I0_init, R0_init,
                 solver, solver_kwargs, k1_init, k2_init, k3_init, T_peak_init, T_max, *, key,  init_par_vmr: float = 0.1,
                 init_sigma_C: float = 1.0, reporting_delay, dt: float = 1.0, underreporting_model="default", **kwargs):
        super().__init__(**kwargs)
        
        underreporting_key, beta_key = jax.random.split(key)

        self.UDE = NeuralUDE(1, width_size, depth, activation, n_freqs, t_scale, population_size,
                             E0_init, I0_init, R0_init,
                             solver=solver, solver_kwargs=solver_kwargs, key=beta_key)
        self.log_k1 = jnp.log(k1_init)

        assert 0.6 < k2_init < 2.5, "k2_init must be in (0.6, 2.5)"
        assert 0.15 < k3_init < 2.0, "k3_init must be in (0.15, 2.0)"
        assert 1 < T_peak_init < 5, "T_peak_init must be in (1, 5)"
        
        k2_shift = (k2_init-0.6)/(2.5-0.6)
        self.logit_k2 = jax.numpy.log(k2_shift / (1 - k2_shift))
        k3_shift = (k3_init-0.15)/(2.0-0.15)
        self.logit_k3 = jax.numpy.log(k3_shift / (1 - k3_shift))
        T_peak_shift = (T_peak_init - 1)/(5 - 1)
        self.logit_T_peak = jax.numpy.log(T_peak_shift / (1 - T_peak_shift))
        self.T_max = T_max
        self.reporting_delay = reporting_delay  # days, delay between infection and reporting
        self.dt = dt

        self.t_scale = t_scale
        self.underreporting_act = activation
        self.underreporting_width = width_size
        self.underreporting_depth = depth

        if underreporting_model == "monotone_increasing":
            self.underreporting_model = MonotoneIncreasingMLP(
                in_size=1, # t as input
                width_size=self.underreporting_width,
                depth=self.underreporting_depth,
                activation=self.underreporting_act,
                final_activation=jax.nn.sigmoid,
                key=underreporting_key,
            )
        elif underreporting_model == "constant":
            assert "not implemented yet"
        else: # default: flexible MLP
            self.underreporting_model = eqx.nn.MLP(
                in_size=1, # t as input
                out_size=1, # state_dim as output
                width_size=self.underreporting_width,
                depth=self.underreporting_depth,
                activation=self.underreporting_act,  # Smooth activation for continuous dynamics
                final_activation=jax.nn.sigmoid,
                key=underreporting_key,
            )

        self.par_vmr = jnp.log(jnp.exp(init_par_vmr) - 1)  # inverse softplus
        self.log_sigma_C = jnp.log(init_sigma_C)
   
    def _beta_scalar(self, t):
        """Scalar time -> scalar beta(t). Keeps autodiff path intact."""
        rhs = self.UDE.rhs
        if len(rhs.freqs) > 0:
            x = rhs.fourier_features(t)                 # shape (2*n_freqs + 1,)
        else:
            x = jnp.array([t])                          # shape (1,)
        return rhs.mlp(x)[0] * rhs.t_scale              # scalar

    def _beta_smoothness(self, ts):
        """∫ (dβ/dt)^2 dt, normalised by interval length for scale invariance."""
        dβ_dt = jax.vmap(jax.grad(lambda τ: self._beta_scalar(τ)))(ts)  # shape (T,)
        return jnp.sum(dβ_dt**2) / (ts[-1] - ts[0]) * 1e-7

    def beta_regularization_loss(self, *, ts=None, mode: str = "test"):
        """
        mode == "L2": classic L2 on beta-MLP params
        mode == "beta_derivative": smoothness penalty on dβ/dt
        mode == "None":          no regularisation
        """
        print(f"Within beta: {mode}")
        if mode == "L2":
            reg = 0.0
            mlp_params, _ = eqx.partition(self.UDE.rhs.mlp, eqx.is_inexact_array)
            for p in jax.tree_util.tree_leaves(mlp_params):
                reg = reg + jnp.sum(p**2)
            return reg

        elif mode == "beta_derivative":
            if ts is None:
                raise ValueError("Provide `ts` (the time grid) for beta-derivative regularisation.")
            return self._beta_smoothness(ts)

        elif mode == "None":
            return 0.0

        else:
            raise ValueError(f"Unknown mode: {mode}")
        
    def _underreporting_scalar(self, t):
        """Scalar time -> scalar underreporting rate ρ(t) in [0,1]. Keeps autodiff path intact."""
        x = jnp.array([t])                           # shape (1,)
        # underreporting_model already has final sigmoid -> value in [0, 1]
        return self.underreporting_model(x)[0]       # scalar

    def _underreporting_smoothness(self, ts):
        """∫ (dρ/dt)^2 dt, normalised by interval length for scale invariance."""
        dρ_dt = jax.vmap(jax.grad(lambda τ: self._underreporting_scalar(τ)))(ts)  # shape (T,)
        return jnp.sum(dρ_dt**2) / (ts[-1] - ts[0]) * 1e-7

    def underreporting_regularization_loss(self, *, ts=None, mode: str = "None"):
        print(f"Within underreporting: {mode}")
        """
        Regularisation for the underreporting MLP.

        mode == "L2":            classic L2 on underreporting-MLP params
        mode == "derivative":    smoothness penalty on dρ/dt (requires `ts`)
        mode == "None":          no regularisation
        """
        if mode == "L2":
            reg = 0.0
            mlp_params, _ = eqx.partition(self.underreporting_model, eqx.is_inexact_array)
            for p in jax.tree_util.tree_leaves(mlp_params):
                reg = reg + jnp.sum(p**2)
            return reg
        elif mode in ("derivative", "underreporting_derivative"):
            if ts is None:
                raise ValueError("Provide `ts` (the time grid) for underreporting-derivative regularisation.")
            return self._underreporting_smoothness(ts)
        elif mode == "None":
            return 0.0
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __call__(self, t_all, t_sub):
        # Call the UDE model to get the state predictions
        y_pred = self.UDE(t_all)[:,:,0]
        
        # Map the E state to concentration using the shedding curve
        dE_new = self.UDE.rhs.beta(t_all) * y_pred[:,0] * y_pred[:,1] /self.UDE.rhs.N
        concentration = map_to_concentration(dE_new, self.T_max, self.logit_T_peak, self.log_k1, self.logit_k2, self.logit_k3, self.dt)

        # get 7-day moving sum of new infections
        underreporting_rate = jax.vmap(self.underreporting_model)(t_sub.reshape(-1, 1))[1:,0]
        underreporting_rate = jnp.concat([underreporting_rate, jnp.repeat(underreporting_rate[-1], len(t_all)-len(t_sub))])  # Extend the underreporting rate to match the length of t_all

        I_new_7d_pred = map_to_7d_sum(y_pred[:, -1], reported_frac=1-underreporting_rate, reporting_delay=self.reporting_delay, dt=self.dt)  # Extract the new infections from the predictions
        eps = 1e-8
        return jnp.log(concentration + eps), I_new_7d_pred  # ensure positivity of log concentrations (we calculate the log of copies/l which is always positive)
        # return jnp.log1p(concentration), I_new_7d_pred # ensure positivity of log concentrations (we calculate the log of copies/l which is always positive)

    def run_model_with_prevalence_output(self, t_all, t_sub):
        # Call the UDE model to get the state predictions
        y_pred = self.UDE(t_all)[:,:,0]

        prevalence = (y_pred[:,1]+y_pred[:,2])/self.UDE.population_size  # Prevalence per 100,000 inhabitants
        
        # Map the E state to concentration using the shedding curve
        dE_new = self.UDE.rhs.beta(t_all) * y_pred[:,0] * y_pred[:,1] /self.UDE.rhs.N
        concentration = map_to_concentration(dE_new, self.T_max, self.logit_T_peak, self.log_k1, self.logit_k2, self.logit_k3, self.dt)
        
        # get 7-day moving sum of new infections
        underreporting_rate = jax.vmap(self.underreporting_model)(t_sub.reshape(-1, 1))[1:,0]
        underreporting_rate = jnp.concat([underreporting_rate, jnp.repeat(underreporting_rate[-1], len(t_all)-len(t_sub))])  # Extend the underreporting rate to match the length of t_all

        I_new_7d_pred = map_to_7d_sum(y_pred[:, -1], reported_frac=1-underreporting_rate, reporting_delay=self.reporting_delay, dt=self.dt)  # Extract the new infections from the predictions
        eps = 1e-8
        return jnp.log(concentration + eps), I_new_7d_pred, prevalence


def moving_window_cumsum(arr, window_size):
    n = arr.shape[0]
    result = jnp.full(n, jnp.nan)

    # Cumulative sum
    cumsum = jnp.cumsum(arr)

    # Compute moving sum by subtracting shifted version
    moving_sum = jnp.where(
        jnp.arange(n) >= window_size - 1,
        cumsum - jnp.pad(cumsum, (window_size, 0), constant_values=0)[:-window_size],
        jnp.nan
    )

    # Fill in result from index `window_size - 1` onward
    result = result.at[window_size - 1:].set(moving_sum[window_size - 1:])
    return result   


def map_to_7d_sum(I_new_pred, reported_frac, reporting_delay, dt):

    I_new_pred = jnp.diff(I_new_pred) # new infections, shape (t_all_dim - 1, y0_dim)
    # implement underreporting
    I_new_pred = I_new_pred * reported_frac # placeholder
    window_steps = int(7.0/dt)
    I_new_7d_pred = moving_window_cumsum(I_new_pred, window_steps) # 7-day moving sum, shape (t_all_dim - 1, y0_dim) with NaNs
    shift_steps = int(reporting_delay / dt)
    # Add a reporting time shift
    def _shift(x, k):
        return jnp.concatenate([jnp.full(k, jnp.nan), x[:-k]]) if k > 0 else x
    return _shift(I_new_7d_pred, shift_steps)  # values for [t_start+1,..., t_end]

def shedding_curve(t, logit_T_peak, log_k1, logit_k2, logit_k3):
    t = jnp.asarray(t)
    log10 = jnp.log(jnp.array(10.0, dtype=t.dtype))

    k1 = jnp.exp(log_k1)
    k2 = jnn.sigmoid(logit_k2) * (2.5 - 0.6) + 0.6
    k3 = jnn.sigmoid(logit_k3) * (2.0 - 0.15) + 0.15
    T_peak = jnn.sigmoid(logit_T_peak) * (5.0 - 1.0) + 1.0

    # Arguments in natural-log space for the 10**(.) terms
    inc_arg  = log10 * (k2 * t)
    peak_arg = log10 * (k2 * T_peak)
    dec_arg  = log10 * (-k3 * (t - T_peak))

    # These are now guaranteed finite (no inf) thanks to _safe_exp
    increase_phase = jnp.exp(jnp.clip(inc_arg, -80.0, 80)) - 1.0          # ≈ 10**(k2*t) - 1
    peak_value     = jnp.exp(jnp.clip(peak_arg, -80.0, 80.0)) - 1.0    # ≈ 10**(k2*T_peak) - 1
    decrease       = jnp.exp(jnp.clip(dec_arg, -80.0, 80.0))         # ≈ 10**(-k3*(t - T_peak))

    w = jnn.sigmoid(10.0 * (t - T_peak))
    return k1 * ((1.0 - w) * increase_phase + w * peak_value * decrease)

def map_to_concentration(E_new, T_max, logit_T_peak, log_k1, logit_k2, logit_k3, dt):
    # dt = 1 # 1 step per day
    s = jnp.arange(0, T_max+dt, dt)
    shedding_values = jax.vmap(shedding_curve, in_axes=(0, None, None, None, None))(s, logit_T_peak, log_k1, logit_k2, logit_k3)
    # conv(E_new, w) matches ∫ E_new(t-s)w(s) ds
    shedding_values = shedding_values*dt # and include Rieman sum
    conc = jax.scipy.signal.fftconvolve(jnp.clip(E_new, a_min=0, a_max=None), shedding_values, mode="valid")
    return conc # values for [t_start+T_max, ..., t_end]


def plot_model(model, t_all, t_phase_1, phase_cut_date,
               I_dates_train, I_train,
               I_dates_val, I_val,
               obs_dates_phase_2, obs_I_phase_2, dates_all,
               conc_dates_train, conc_train,
               conc_dates_val, conc_val, t_all_idx, conc_y_axis_label="Log flow normalized\nconcentration [log(copies/l)]",
               conc_dates_test=[], conc_test=[]):
    
    from scipy.stats import nbinom
    pred_conc, I_new_7d_pred = model(t_all, t_phase_1)

    vmr = 1 + jnn.softplus(model.par_vmr)
    sigma_C = jnp.exp(model.log_sigma_C)

    fig, ax = plt.subplots(figsize=(5.5, 3), dpi=300)
    ax.scatter(I_dates_train, I_train, label="Training", color="goldenrod", s=10)
    ax.scatter(I_dates_val, I_val, label="Validation", color="black", s=10)
    ax.scatter(obs_dates_phase_2, obs_I_phase_2, label="Test", color="#595959", s=10, alpha=0.8)

    ax.axvline(pd.to_datetime(phase_cut_date), color="#595959", linestyle='--')

    ax.plot(dates_all[1:], I_new_7d_pred, label="Prediction", color="#8B0000")

    # Error band: alpha = 0.05
    alpha = 0.05
    eps = 1e-12
    vmr = jnp.maximum(vmr, 1.0 + 1e-6)   # keep away from exactly 1 to avoid r→∞
    p = 1.0 / vmr
    r = I_new_7d_pred / (vmr - 1.0)
    # Clip to numerically safe range
    p = jnp.clip(p, eps, 1.0 - eps)
    r = jnp.clip(r, eps, 1e12)
    q_lo = nbinom.ppf(alpha/2.0, r, p)
    q_hi = nbinom.ppf(1.0 - alpha/2.0, r, p)
    ax.fill_between(dates_all[1:], 
                    q_lo, 
                    q_hi, 
                    color="#8B0000", alpha=0.2, label="95% prediction interval")

    ax.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 5, 9)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
    plt.ylabel("7-day moving sum of\nnew infections [#]")
    plt.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(5.5, 3), dpi=300)
    ax2.scatter(conc_dates_train, conc_train, label="Training", color="royalblue", s=10)
    ax2.scatter(conc_dates_val, conc_val, label="Validation", color="black", s=10)
    if len(conc_dates_test) > 0:
        ax2.scatter(conc_dates_test, conc_test, label="Test", color="#595959", s=10)

    ax2.plot(dates_all[t_all_idx*model.dt >= int(model.T_max)], pred_conc, label="Prediction", color="#8B0000")

    # Error band: ±1 std
    ax2.fill_between(dates_all[t_all_idx*model.dt >= int(model.T_max)],
                     pred_conc - 1.96*sigma_C,
                     pred_conc + 1.96*sigma_C,
                     color="#8B0000", alpha=0.2, label="95% prediction interval")

    ax2.legend()
    ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 5, 9)))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
    ax2.axvline(pd.to_datetime(phase_cut_date), color="#595959", linestyle='--')
    plt.ylabel(conc_y_axis_label)
    plt.tight_layout()
    plt.close(fig)
    plt.close(fig2)
    return fig, fig2


def plot_model_zero_inflated_noise_model(model, t_all, t_phase_1, phase_cut_date,
               I_dates_train, I_train,
               I_dates_val, I_val,
               obs_dates_phase_2, obs_I_phase_2, dates_all,
               conc_dates_train, conc_train,
               conc_dates_val, conc_val, t_all_idx, conc_y_axis_label="Flow normalized\nconcentration [log(copies/l)]",
               conc_dates_test=[], conc_test=[]):
    
    from scipy.stats import nbinom
    pred_conc, I_new_7d_pred = model.predict_without_log_transform(t_all, t_phase_1)

    vmr = 1 + jnn.softplus(model.par_vmr)
    sigma_C = jnp.exp(model.log_sigma_C)

    fig, ax = plt.subplots(figsize=(5.5, 3), dpi=300)
    ax.scatter(I_dates_train, I_train, label="Training", color="goldenrod", s=10)
    ax.scatter(I_dates_val, I_val, label="Validation", color="black", s=10)
    ax.scatter(obs_dates_phase_2, obs_I_phase_2, label="Test", color="#595959", s=10, alpha=0.8)

    ax.axvline(pd.to_datetime(phase_cut_date), color="#595959", linestyle='--')

    ax.plot(dates_all[1:], I_new_7d_pred, label="Prediction", color="#8B0000")

    # Error band: alpha = 0.05
    alpha = 0.05
    eps = 1e-12
    vmr = jnp.maximum(vmr, 1.0 + 1e-6)   # keep away from exactly 1 to avoid r→∞
    p = 1.0 / vmr
    r = I_new_7d_pred / (vmr - 1.0)
    # Clip to numerically safe range
    p = jnp.clip(p, eps, 1.0 - eps)
    r = jnp.clip(r, eps, 1e12)
    q_lo = nbinom.ppf(alpha/2.0, r, p)
    q_hi = nbinom.ppf(1.0 - alpha/2.0, r, p)
    ax.fill_between(dates_all[1:], 
                    q_lo, 
                    q_hi, 
                    color="#8B0000", alpha=0.2, label="95% prediction interval")

    ax.legend()
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 5, 9)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
    plt.ylabel("7-day moving sum of\nnew infections [#]")
    plt.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(5.5, 3), dpi=300)
    ax2.scatter(conc_dates_train, conc_train, label="Training", color="royalblue", s=10)
    ax2.scatter(conc_dates_val, conc_val, label="Validation", color="black", s=10)
    if len(conc_dates_test) > 0:
        ax2.scatter(conc_dates_test, conc_test, label="Test", color="#595959", s=10)

    ax2.plot(dates_all[t_all_idx*model.dt >= int(model.T_max)], pred_conc, label="Prediction", color="#8B0000")

    # Error band: ±1 std
    ax2.fill_between(dates_all[t_all_idx*model.dt >= int(model.T_max)],
                     pred_conc - 1.96*sigma_C,
                     pred_conc + 1.96*sigma_C,
                     color="#8B0000", alpha=0.2, label="95% prediction interval")

    ax2.legend()
    ax2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 5, 9)))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
    ax2.axvline(pd.to_datetime(phase_cut_date), color="#595959", linestyle='--')
    # ax2.set_yscale("log")
    plt.ylabel(conc_y_axis_label)
    plt.tight_layout()
    plt.close(fig)
    plt.close(fig2)
    return fig, fig2


def plot_shedding_curve(model):
    s = jnp.arange(0, model.T_max+model.dt, model.dt)
    shedding_values = jax.vmap(shedding_curve, in_axes=(0, None, None, None, None))(s, model.logit_T_peak, model.log_k1, model.logit_k2, model.logit_k3)

    k1 = jnp.exp(model.log_k1)
    k2 = jnn.sigmoid(model.logit_k2)*(2.5-0.6) + 0.6
    k3 = jnn.sigmoid(model.logit_k3)*(2.0-0.15) + 0.15
    T_peak = jnn.sigmoid(model.logit_T_peak)*(5-1)+1
    fig, ax = plt.subplots(ncols=1, figsize=(5.5, 3), dpi=300)
    ax.plot(s, shedding_values, color='goldenrod', label=f"k1={k1:.2f},\nk2={k2:.2f},\nk3={k3:.2f},\nT_peak={T_peak:.1f}d")
    ax.set_xlabel('Days since infection')
    ax.set_ylabel('Shedding intensity')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_SEIR_prediction(model, t_all, dates_all, phase_cut_date):
    pred = model.UDE(t_all)[:,:,0]

    fig, axs = plt.subplots(ncols=4, figsize=(9, 2.5), dpi=300)
    axs[0].plot(dates_all, pred[:,0], c="saddlebrown")
    axs[0].set_title(r"$S$")
    axs[1].plot(dates_all, pred[:,1], c="peru")
    axs[1].set_title(r"$E$")
    axs[2].plot(dates_all, pred[:,2], "darkgoldenrod")
    axs[2].set_title(r"$I$")
    axs[3].plot(dates_all, pred[:,3], c="goldenrod")
    axs[3].set_title(r"$R$")

    for i, ax in enumerate(axs):
        # enforce scientific notation on y-axis
        formatter = mticker.ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-3, 3))  # always use scientific notation outside this range
        ax.yaxis.set_major_formatter(formatter)
        if i ==0:
            ax.set_ylabel("Compartment size [#]")
        ax.tick_params(axis='x', rotation=45)
        ax = plt.gca()


        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
        axs[i].xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 5, 9)))
        axs[i].axvline(pd.to_datetime(phase_cut_date), color="#595959", linestyle='--')
    plt.tight_layout()
    plt.close(fig)
    return fig

def plot_test_positive_rate(model, t_all, dates_all, prev_phase_cut_date, town, data):
    pred = model.UDE(t_all)[:,:,0]
    prevalence = (pred[:,1]+pred[:,2])/model.UDE.rhs.N * 100.0  # Prevalence in percent
    se = 0.83
    sp = 1.00
    pos_rate = se*prevalence + (1.0-sp)*(1-prevalence)

    fig, ax = plt.subplots(ncols=1, figsize=(4.5, 2.5), dpi=300)
    ax.plot(dates_all, pos_rate, c="saddlebrown", label="Integrative model")
    ax.scatter(data["prevalence_dates_train"], data["pos_tests_train"]/data["n_tests_train"]*100, color="#63A066", s=10, label="Training")
    ax.scatter(data["prevalence_dates_val"], data["pos_tests_val"]/data["n_tests_val"]*100, color="#043507", s=10, label="Validation")
    ax.scatter(data["prevalence_dates_test"], data["pos_tests_test"]/data["n_tests_test"]*100, color="#595959", s=10, label="Test")
    ax.legend()
    ax.set_ylabel("Test positivity [%]")
    ax.tick_params(axis='x', rotation=45)
    ax = plt.gca()
    # ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 5, 9)))   # Jan, Mar, May, …
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
    ax.axvline(pd.to_datetime(prev_phase_cut_date), color="#595959", linestyle='--')
    plt.title(f"{town}")
    plt.tight_layout()
    plt.close(fig)
    return fig

def plot_beta_prediction(model, t_all, dates_all, phase_cut_date):    
    if len(model.UDE.rhs.freqs)>0:
        beta_pred = jax.vmap(model.UDE.rhs.mlp)(jax.vmap(model.UDE.rhs.fourier_features)(t_all.reshape(-1,1)))
    else:
        beta_pred = jax.vmap(model.UDE.rhs.mlp)(t_all.reshape(-1,1))

    beta_pred = beta_pred[:,0]

    fig, ax = plt.subplots(ncols=1, figsize=(4.5, 2.5), dpi=300)
    ax.plot(dates_all, beta_pred, c="saddlebrown")
    ax.set_ylabel(r"$\beta$")
    ax.tick_params(axis='x', rotation=45)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 5, 9)))   # Jan, Mar, May, …
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
    ax.axvline(pd.to_datetime(phase_cut_date), color="#595959", linestyle='--')
    plt.tight_layout()
    plt.close(fig)
    return fig

def plot_effective_reproduction_number(model, t_all, dates_all, phase_cut_date):
    pred = model.UDE(t_all)[:,:,0]
    if len(model.UDE.rhs.freqs)>0:
        beta_pred = jax.vmap(model.UDE.rhs.mlp)(jax.vmap(model.UDE.rhs.fourier_features)(t_all.reshape(-1,1)))
    else:
        beta_pred = jax.vmap(model.UDE.rhs.mlp)(t_all.reshape(-1,1))

    beta_pred = beta_pred[:,0]
    S = pred[:,0]
    gamma = 0.2

    R_t = beta_pred/gamma * S/model.UDE.rhs.N

    fig, ax = plt.subplots(ncols=1, figsize=(4.5, 2.5), dpi=300)
    ax.plot(dates_all, R_t, c="saddlebrown")
    ax.set_ylabel(r"$R_t$")
    ax = plt.gca()
    ax.tick_params(axis='x', rotation=45)
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 5, 9)))   # Jan, Mar, May, …
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
    ax.axvline(pd.to_datetime(phase_cut_date), color="#595959", linestyle='--')
    ax.axhline(1.0, color="#A1A1A1", linestyle='--')
    plt.tight_layout()
    plt.close(fig)
    return fig


def plot_underreporting(model, t_all, t_phase_1, dates_all, phase_cut_date):
    underreporting_rate = jax.vmap(model.underreporting_model)(t_phase_1.reshape(-1, 1))[:,0]
    underreporting_rate = jnp.concat([underreporting_rate, jnp.repeat(underreporting_rate[-1], len(t_all)-len(t_phase_1))])  # Extend the underreporting rate to match the length of t_all

    fig, ax = plt.subplots(ncols=1, figsize=(4.5, 2.5), dpi=300)
    ax.plot(dates_all, (1-underreporting_rate)*100)
    ax.set_ylabel("Reporting rate [%]")
    ax.tick_params(axis='x', rotation=45)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 5, 9)))   # Jan, Mar, May, …
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
    ax.axvline(pd.to_datetime(phase_cut_date), color="#595959", linestyle='--')
    #ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.close(fig)
    return fig