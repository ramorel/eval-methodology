"""
2PL Bayesian IRT — 45 Person × 50 Item Design
===============================================
Fits a two-parameter logistic IRT model on the full
3 (model) × 3 (scaffold) × 5 (seed) crossed design.

Persons = model × scaffold × seed combinations.
Items = TruthfulQA questions.

After fitting, extracts posterior summaries and
examines how ability estimates decompose by
model and scaffold.
"""

import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az
import matplotlib.pyplot as plt

# Load and shape data
df = pd.read_csv("results.csv")

# Person = model × scaffold × seed
df["person_id"] = df["model"] + "_" + df["scaffold"] + "_" + df["run_id"]

# Create person and item indices
persons = sorted(df["person_id"].unique())
items = sorted(df["sample_id"].unique())
person_to_idx = {p: i for i, p in enumerate(persons)}
item_to_idx = {it: i for i, it in enumerate(items)}

df["person_idx"] = df["person_id"].map(person_to_idx)
df["item_idx"] = df["sample_id"].map(item_to_idx)

N = len(persons)
J = len(items)
print(f"Persons: {N}, Items: {J}")

# Build response matrix (N × J)
response_matrix = np.full((N, J), np.nan)
for _, row in df.iterrows():
    pi = person_to_idx[row["person_id"]]
    ji = item_to_idx[row["sample_id"]]
    response_matrix[pi, ji] = row["score"]
 
# Check for missing
n_missing = np.isnan(response_matrix).sum()
print(f"Missing cells: {n_missing} of {N * J}")
 
# Convert to flat arrays for NumPyro (drop any missing)
person_idx = []
item_idx = []
y = []
for i in range(N):
    for j in range(J):
        if not np.isnan(response_matrix[i, j]):
            person_idx.append(i)
            item_idx.append(j)
            y.append(int(response_matrix[i, j]))
 
person_idx = jnp.array(person_idx)
item_idx = jnp.array(item_idx)
y = jnp.array(y)
print(f"Total observations: {len(y)}")

# ── 2. Model specification ──────────────────────────────────────────────────
def irt_2pl(person_idx, item_idx, y=None, N=None, J=None):
    """Two-parameter logistic IRT model.
    
    θ_i ~ N(0, 1)           ability (identified by unit variance)
    b_j ~ N(0, 2)           difficulty
    log(a_j) ~ N(0, 0.5)    discrimination (log-normal)
    P(y=1) = sigmoid(a_j * (θ_i - b_j))
    """
    # Person abilities
    theta = numpyro.sample("theta", dist.Normal(0, 1).expand([N]))
    
    # Item parameters
    b = numpyro.sample("b", dist.Normal(0, 2).expand([J]))
    log_a = numpyro.sample("log_a", dist.Normal(0, 0.5).expand([J]))
    a = numpyro.deterministic("a", jnp.exp(log_a))
    
    # Likelihood
    logit_p = a[item_idx] * (theta[person_idx] - b[item_idx])
    numpyro.sample("obs", dist.Bernoulli(logits=logit_p), obs=y)
 
# ── 3. Fit ───────────────────────────────────────────────────────────────────
print("\nFitting 2PL IRT model...")
print("This may take a few minutes on CPU.\n")

kernel = NUTS(irt_2pl)
mcmc = MCMC(
    kernel,
    num_warmup=1000,
    num_samples=2000,
    num_chains=2,
    progress_bar=True,
)
 
rng_key = jax.random.PRNGKey(42)
mcmc.run(
    rng_key,
    person_idx=person_idx,
    item_idx=item_idx,
    y=y,
    N=N,
    J=J,
)

# ── 4. Diagnostics ──────────────────────────────────────────────────────────
 
print("\n" + "=" * 70)
print("MCMC DIAGNOSTICS")
print("=" * 70)
 
# Convert to ArviZ
idata = az.from_numpyro(mcmc)
 
# Summary for key parameters
print("\nθ (ability) summary:")
theta_summary = az.summary(idata, var_names=["theta"], round_to=3)
print(theta_summary[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat", "ess_bulk"]].to_string())
 
print("\nItem parameter summary (first 10):")
item_summary = az.summary(idata, var_names=["a", "b"], round_to=3)
print(item_summary[["mean", "sd", "hdi_3%", "hdi_97%", "r_hat"]].head(20).to_string())
 
# Check for divergences
n_div = idata.sample_stats["diverging"].sum().item()
print(f"\nDivergences: {n_div}")
 
# ── 5. Posterior analysis: decompose θ by model and scaffold ────────────────
 
print("\n" + "=" * 70)
print("ABILITY ESTIMATES BY MODEL × SCAFFOLD")
print("=" * 70)
 
theta_means = mcmc.get_samples()["theta"].mean(axis=0)
theta_sds = mcmc.get_samples()["theta"].std(axis=0)
 
# Build a DataFrame of person-level results
person_results = []
for person_id, idx in person_to_idx.items():
    parts = person_id.split("_")
    # Model name has hyphens, scaffold and run_id are single tokens
    # Format: claude-haiku-4-5_baseline_run1
    # Split on last two underscores
    rsplit = person_id.rsplit("_", 2)
    model = rsplit[0]
    scaffold = rsplit[1]
    seed = rsplit[2]
    
    person_results.append({
        "person_id": person_id,
        "model": model,
        "scaffold": scaffold,
        "seed": seed,
        "theta_mean": float(theta_means[idx]),
        "theta_sd": float(theta_sds[idx]),
        "raw_score": float(response_matrix[idx].mean()),
    })
 
pr_df = pd.DataFrame(person_results)
 
# Model × Scaffold means
print("\nPosterior θ means by Model × Scaffold:")
theta_table = pr_df.pivot_table(
    values="theta_mean", index="model", columns="scaffold", aggfunc="mean"
)
print(theta_table.round(3))
 
print("\nPosterior θ SD (across seeds) by Model × Scaffold:")
theta_sd_table = pr_df.pivot_table(
    values="theta_mean", index="model", columns="scaffold", aggfunc="std"
)
print(theta_sd_table.round(3))
 
print("\nModel marginal θ means:")
print(pr_df.groupby("model")["theta_mean"].agg(["mean", "std"]).round(3))
 
print("\nScaffold marginal θ means:")
print(pr_df.groupby("scaffold")["theta_mean"].agg(["mean", "std"]).round(3))
 
# ── 6. Item analysis ────────────────────────────────────────────────────────
 
print("\n" + "=" * 70)
print("ITEM PARAMETERS")
print("=" * 70)
 
a_means = mcmc.get_samples()["a"].mean(axis=0)
b_means = mcmc.get_samples()["b"].mean(axis=0)
a_sds = mcmc.get_samples()["a"].std(axis=0)
 
# Get question text for each item
item_text = {}
for _, row in df.drop_duplicates("sample_id").iterrows():
    item_text[row["sample_id"]] = row["question"]
 
item_results = []
for item_id, idx in item_to_idx.items():
    item_results.append({
        "sample_id": item_id,
        "question": item_text.get(item_id, "")[:60],
        "a_mean": float(a_means[idx]),
        "a_sd": float(a_sds[idx]),
        "b_mean": float(b_means[idx]),
        "raw_p": float(response_matrix[:, idx].mean()),
    })
 
ir_df = pd.DataFrame(item_results).sort_values("a_mean", ascending=False)
 
print("\nTop 10 most discriminating items:")
print(ir_df.head(10)[["sample_id", "a_mean", "a_sd", "b_mean", "raw_p", "question"]].to_string(index=False))
 
print("\nBottom 10 least discriminating items:")
print(ir_df.tail(10)[["sample_id", "a_mean", "a_sd", "b_mean", "raw_p", "question"]].to_string(index=False))
 
# Count informative items (discrimination clearly above prior mean ~1.0)
n_informative = (ir_df["a_mean"] > 1.3).sum()
n_uninformative = (ir_df["a_mean"] < 0.8).sum()
print(f"\nHighly discriminating items (a > 1.3): {n_informative}")
print(f"Low discrimination items (a < 0.8): {n_uninformative}")
print(f"Items near prior (0.8 < a < 1.3): {len(ir_df) - n_informative - n_uninformative}")
 
# ── 7. Save results ─────────────────────────────────────────────────────────
 
pr_df.to_csv("person_irt_results.csv", index=False)
ir_df.to_csv("item_irt_results.csv", index=False)
print("\nSaved: person_irt_results.csv, item_irt_results.csv")
 