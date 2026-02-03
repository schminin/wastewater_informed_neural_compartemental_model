import json
import equinox as eqx
import diffrax
import jax
import jax.nn as jnn

activation_fct_mapper = {
        "tanh": jnn.tanh,
        "sigmoid": jnn.sigmoid,
        "gelu": jnn.gelu,
        "silu": jnn.silu,
    }

solver_mapper = {
        "Tsit5": diffrax.Tsit5(),
        "Dopri5": diffrax.Dopri5(),
        # add others as needed
    }

# Storage functionalities
def save_model(model, experiment_log_dir, trial=1):
    # Save parameter leaves (weights, learnable arrays)
    eqx.tree_serialise_leaves(f"{experiment_log_dir}/model_{trial}.eqx", model)


def load_model(model_class, experiment_log_dir):
    # Load static config
    with open(f"{experiment_log_dir}/model_config.json", "r") as f:
        config = json.load(f)

    solver = solver_mapper[config["solver"]]

    # Recreate template model
    key = jax.random.PRNGKey(config["seed"])
    model_template = model_class(
        state_dim=config["state_dim"],
        width_size=config["width_size"],
        depth=config["depth"],
        key=key,
        solver=solver,
        solver_kwargs=config["solver_kwargs"],
    )

    # Load leaves into the template
    model = eqx.tree_deserialise_leaves(f"{experiment_log_dir}/model.eqx", model_template)
    return model, config

