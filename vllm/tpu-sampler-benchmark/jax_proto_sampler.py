import jax
import jax.numpy as jnp
from jax import random, jit
from time import time
import jax
import jax.numpy as jnp
from jax import random
from typing import Optional, Dict


def forward_native(
    logits: jnp.ndarray,
    k: Optional[jnp.ndarray],
    p: Optional[jnp.ndarray],
    generators: Dict[int, random.PRNGKey]=None,
) -> jnp.ndarray:
    """JAX implementation of top-k and top-p sampling."""
    logits = apply_top_k_top_p(logits, k, p)
    probs = jax.nn.softmax(logits, axis=-1).astype(jnp.float32)
    return random_sample(probs, generators)

def apply_top_k_top_p(
    logits: jnp.ndarray,
    k: Optional[jnp.ndarray],
    p: Optional[jnp.ndarray],
) -> jnp.ndarray:
    """Apply top-k and top-p masks to the logits."""
    if k is None and p is None:
        return logits

    logits_sort = jnp.sort(logits, axis=-1)
    logits_idx = jnp.argsort(logits, axis=-1)

    if k is not None:
        # Apply top-k.
        top_k_mask = logits_sort.shape[1] - k  # shape: B
        top_k_mask = logits_sort[jnp.arange(logits_sort.shape[0]), top_k_mask]
        top_k_mask = logits_sort < top_k_mask[:, None]
        logits_sort = jnp.where(top_k_mask, -jnp.inf, logits_sort)

    if p is not None:
        # Apply top-p.
        probs_sort = jax.nn.softmax(logits_sort, axis=-1)
        probs_sum = jnp.cumsum(probs_sort, axis=-1)
        top_p_mask = probs_sum <= 1 - p[:, None]
        top_p_mask = top_p_mask.at[:, -1].set(False)
        logits_sort = jnp.where(top_p_mask, -jnp.inf, logits_sort)

    # Re-sort the probabilities.
    logits = logits_sort[jnp.arange(logits_sort.shape[0])[:, None], logits_idx]
    return logits

def random_sample(
    probs: jnp.ndarray,
    generators: Dict[int, random.PRNGKey],
) -> jnp.ndarray:
    """Randomly sample from the probabilities."""
    q = random.exponential(random.PRNGKey(0), shape=probs.shape)
    if generators:
        for i, key in generators.items():
            q = q.at[i].set(random.exponential(key, shape=probs.shape[1]))
    return jnp.argmax(probs / q, axis=-1).reshape(-1)

# Make sure TPU is listed here
print(jax.devices())
V = 128256
H = V // 32

# Initialize linear layer
key = jax.random.PRNGKey(42)
key_W, key_b = jax.random.split(key)
W = jax.random.normal(key_W, (H, V))
b = jax.random.normal(key_b, (V,))


@jit
def linear(W, b, x):
    return jnp.dot(x, W) + b


@jit
def model_with_sample(x, k, p)->jnp.ndarray:
    y = linear(W, b, x)
    y = forward_native(y, k, p)
    return y

# Some sampling metadata for tracing
topp = jnp.array([0.6])
topk = jnp.array([12], dtype=jnp.int32)

# Compile graph, keep consistency with xla benchmark code 
for B in [1, 4, 16, 32]:
    x = random.normal(random.PRNGKey(0), (B, H))
    s = time()
    out = model_with_sample(x, topk, topp).block_until_ready()
    print(f"Compiling/Warmup {B}", time()-s)

# Run
for B in [1, 4, 16, 32]:
    x = random.normal(random.PRNGKey(0), (B, H))
    for _ in range(4):
        s = time()
        out = model_with_sample(x, topk, topp).block_until_ready()
        print(f"Running {B}", time()-s)
