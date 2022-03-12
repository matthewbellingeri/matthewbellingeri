import jax
import jax.numpy as jnp

def policy_gradient_loss(params, apply_fn, states, actions, rewards):
    log_probs = jnp.log(apply_fn(params, states))
    action_log_probs = log_probs[jnp.arange(len(actions)), actions]
    return -jnp.mean(action_log_probs * rewards)

print("RL Policy Gradient defined with JAX.")