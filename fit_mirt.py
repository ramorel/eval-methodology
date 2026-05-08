"""
Multidimensional IRT (MIRT) — 4-Dimensional Compensatory Model via SVI
========================================================================
120 Persons (model × scaffold/persona × seed) × 35 informative Items × 4 Dimensions

Uses Stochastic Variational Inference (AutoNormal guide) instead of MCMC.
This avoids the rotational indeterminacy that prevents NUTS from mixing,
and matches the estimation approach used by the `mirt` R package (EM/quasi-Newton).

Model specification (unchanged):
  θ_i ~ MVN(0, I_4)
  a_jd ~ N(0, 2)
  d_j ~ N(0, 3)
  logit P(y=1) = Σ_d [a_jd * θ_id] + d_j
"""

import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, autoguide, Predictive
import optax

# ── 1. Load and reshape ─────────────────────────────────────────────────────

df = pd.read_csv("results.csv")
df["person_id"] = df["model"] + "|" + df["scaffold"] + "|" + df["run_id"]

persons = sorted(df["person_id"].unique())
items = sorted(df["sample_id"].unique())
person_to_idx = {p: i for i, p in enumerate(persons)}
item_to_idx = {it: i for i, it in enumerate(items)}

N = len(persons)
J = len(items)
D = 4

print(f"Persons: {N}, Items: {J}, Dimensions: {D}")

# Build response matrix
response_matrix = np.full((N, J), np.nan)
for _, row in df.iterrows():
    pi = person_to_idx[row["person_id"]]
    ji = item_to_idx[row["sample_id"]]
    response_matrix[pi, ji] = row["score"]

# Drop items with zero variance
item_vars = np.nanvar(response_matrix, axis=0)
informative_mask = item_vars > 0
print(f"Items with variance: {informative_mask.sum()} of {J}")

# Keep track of original item IDs for the informative items
informative_item_ids = [it for it, m in zip(items, informative_mask) if m]
response_matrix = response_matrix[:, informative_mask]
items = informative_item_ids
item_to_idx = {it: i for i, it in enumerate(items)}
J = len(items)

# Build flat observation arrays
person_idx, item_idx_flat = np.where(~np.isnan(response_matrix))
y = response_matrix[person_idx, item_idx_flat].astype(int)

person_idx = jnp.array(person_idx)
item_idx = jnp.array(item_idx_flat)
y = jnp.array(y)

print(f"Informative items: {J}")
print(f"Total observations: {len(y)}")
print(f"Parameters: {N*D} abilities + {J*D} loadings + {J} intercepts = {N*D + J*D + J}")
print()

# ── 2. Model specification ──────────────────────────────────────────────────

def mirt_model(person_idx, item_idx, y=None, N=None, J=None, D=4):
    # Person abilities — MVN(0, I)
    theta = numpyro.sample(
        "theta",
        dist.Normal(0, 1).expand([N, D]),
    )

    # Item loadings
    a = numpyro.sample(
        "a",
        dist.Normal(0, 2).expand([J, D]),
    )

    # Item intercepts
    d = numpyro.sample(
        "d",
        dist.Normal(0, 3).expand([J]),
    )

    # Logit
    logit_p = jnp.sum(a[item_idx] * theta[person_idx], axis=1) + d[item_idx]

    # Likelihood
    numpyro.sample("obs", dist.Bernoulli(logits=logit_p), obs=y)


# ── 3. Fit via SVI ──────────────────────────────────────────────────────────

print("Fitting 4-dimensional MIRT via SVI (variational inference)...")
print()

# AutoNormal learns a diagonal normal approximation for each parameter
guide = autoguide.AutoNormal(mirt_model)

# Adam optimizer with learning rate schedule: start higher, decay
scheduler = optax.exponential_decay(
    init_value=0.01,
    transition_steps=1000,
    decay_rate=0.5,
    staircase=True,
)
optimizer = optax.adam(scheduler)

svi = SVI(mirt_model, guide, optimizer, loss=Trace_ELBO())

rng_key = jax.random.PRNGKey(42)
n_steps = 10000

svi_state = svi.init(
    rng_key,
    person_idx=person_idx,
    item_idx=item_idx,
    y=y,
    N=N,
    J=J,
    D=D,
)

# Training loop with loss tracking
losses = []
print_every = 1000

for step in range(n_steps):
    svi_state, loss = svi.update(
        svi_state,
        person_idx=person_idx,
        item_idx=item_idx,
        y=y,
        N=N,
        J=J,
        D=D,
    )
    losses.append(float(loss))

    if (step + 1) % print_every == 0:
        avg_loss = np.mean(losses[-print_every:])
        print(f"  Step {step+1:>6}/{n_steps}: ELBO loss = {avg_loss:.1f}")

# Check convergence: compare last 1000 to previous 1000
final_loss = np.mean(losses[-1000:])
prev_loss = np.mean(losses[-2000:-1000])
pct_change = abs(final_loss - prev_loss) / abs(prev_loss) * 100
print(f"\n  Final avg loss: {final_loss:.1f}")
print(f"  Loss change (last 2k steps): {pct_change:.2f}%")
if pct_change > 1.0:
    print("  WARNING: Loss may not have converged. Consider more steps.")

# ── 4. Extract parameters ───────────────────────────────────────────────────

print()
print("=" * 70)
print("EXTRACTING VARIATIONAL PARAMETERS")
print("=" * 70)

params = svi.get_params(svi_state)
median_params = guide.median(params)

theta_means = np.array(median_params["theta"])  # (N, D)
a_means = np.array(median_params["a"])  # (J, D)
d_means = np.array(median_params["d"])  # (J,)

# Get variational SDs from the guide
# AutoNormal stores scale parameters
quantiles = guide.quantiles(params, [0.025, 0.975])
theta_lo = np.array(quantiles["theta"][0])
theta_hi = np.array(quantiles["theta"][1])
a_lo = np.array(quantiles["a"][0])
a_hi = np.array(quantiles["a"][1])

# Approximate SD from 95% CI
theta_sds = (theta_hi - theta_lo) / 3.92
a_sds = (a_hi - a_lo) / 3.92

print(f"θ range: [{theta_means.min():.2f}, {theta_means.max():.2f}]")
print(f"a range: [{a_means.min():.2f}, {a_means.max():.2f}]")
print(f"d range: [{d_means.min():.2f}, {d_means.max():.2f}]")

# ── 5. Ability estimates by model × scaffold ────────────────────────────────

print()
print("=" * 70)
print("ABILITY ESTIMATES BY MODEL × SCAFFOLD/PERSONA")
print("=" * 70)

# Build person-level results
person_results = []
for person_id, idx in person_to_idx.items():
    parts = person_id.split("|")
    model = parts[0]
    scaffold = parts[1]
    seed = parts[2]

    row = {
        "person_id": person_id,
        "model": model,
        "scaffold": scaffold,
        "seed": seed,
        "raw_score": float(np.nanmean(response_matrix[idx])),
    }
    for dim in range(D):
        row[f"theta_d{dim+1}_mean"] = float(theta_means[idx, dim])
        row[f"theta_d{dim+1}_sd"] = float(theta_sds[idx, dim])
    person_results.append(row)

pr_df = pd.DataFrame(person_results)

for dim in range(D):
    col = f"theta_d{dim+1}_mean"
    print(f"\nDimension {dim+1} — θ means by Model × Scaffold/Persona:")
    pivot = pr_df.pivot_table(values=col, index="model", columns="scaffold", aggfunc="mean")
    print(pivot.round(3))

    model_means = pr_df.groupby("model")[col].mean()
    scaffold_means = pr_df.groupby("scaffold")[col].mean()
    print(f"  Model range: {model_means.max() - model_means.min():.3f}")
    print(f"  Scaffold range: {scaffold_means.max() - scaffold_means.min():.3f}")

# ── 6. Item loading structure ────────────────────────────────────────────────

print()
print("=" * 70)
print("ITEM LOADINGS (Variational Posterior Means)")
print("=" * 70)

item_meta = df.drop_duplicates("sample_id").set_index("sample_id")

print(f"\n{'ID':>4}", end="")
for dim in range(D):
    print(f" {'D'+str(dim+1):>7}", end="")
print(f" {'d':>7} {'p':>5} {'Category':<22} {'Question':<50}")
print("-" * (4 + D * 8 + 8 + 6 + 22 + 50))

# Sort by magnitude of largest loading
max_loading = np.max(np.abs(a_means), axis=1)
sort_idx = np.argsort(-max_loading)

for idx in sort_idx:
    item_id = items[idx]
    cat = item_meta.loc[item_id, "category"] if item_id in item_meta.index else ""
    q = item_meta.loc[item_id, "question"][:48] if item_id in item_meta.index else ""
    raw_p = np.nanmean(response_matrix[:, idx])

    print(f"{item_id:>4}", end="")
    for dim in range(D):
        val = a_means[idx, dim]
        marker = "*" if abs(val) > 0.5 else " "
        print(f" {val:>6.2f}{marker}", end="")
    print(f" {d_means[idx]:>7.2f} {raw_p:>5.2f} {cat:<22} {q:<50}")

# ── 7. Loading structure summary ────────────────────────────────────────────

print()
print("=" * 70)
print("LOADING STRUCTURE SUMMARY")
print("=" * 70)

for dim in range(D):
    strong = [(items[j], a_means[j, dim]) for j in range(J) if abs(a_means[j, dim]) > 0.5]
    strong.sort(key=lambda x: -abs(x[1]))
    print(f"\nDimension {dim+1} — items with |loading| > 0.5:")
    if strong:
        for item_id, val in strong:
            cat = item_meta.loc[item_id, "category"] if item_id in item_meta.index else ""
            q = item_meta.loc[item_id, "question"][:50] if item_id in item_meta.index else ""
            print(f"  {item_id:>4}  a={val:>6.2f}  {cat:<22} {q}")
    else:
        print("  (none)")

# ── 8. Posterior predictive check ────────────────────────────────────────────

print()
print("=" * 70)
print("POSTERIOR PREDICTIVE CHECK")
print("=" * 70)

logit_pred = jnp.sum(a_means[item_idx] * theta_means[person_idx], axis=1) + d_means[item_idx]
p_pred = jax.nn.sigmoid(logit_pred)

y_pred = (np.array(p_pred) > 0.5).astype(int)
accuracy = (y_pred == np.array(y)).mean()
print(f"Classification accuracy: {accuracy:.3f}")

# By model
for model_name in sorted(df["model"].unique()):
    model_persons = [person_to_idx[p] for p in persons if p.startswith(model_name + "|")]
    mask = np.isin(np.array(person_idx), model_persons)
    model_acc = (y_pred[mask] == np.array(y)[mask]).mean()
    model_obs_mean = np.array(y)[mask].mean()
    model_pred_mean = np.array(p_pred)[mask].mean()
    print(f"  {model_name}: acc={model_acc:.3f}, obs={model_obs_mean:.3f}, pred={model_pred_mean:.3f}")

# By scaffold/persona
print()
for scaffold in sorted(df["scaffold"].unique()):
    scaffold_persons = [person_to_idx[p] for p in persons if f"|{scaffold}|" in p]
    mask = np.isin(np.array(person_idx), scaffold_persons)
    if mask.sum() > 0:
        obs = np.array(y)[mask].mean()
        pred = np.array(p_pred)[mask].mean()
        print(f"  {scaffold:<25} obs={obs:.3f}, pred={pred:.3f}")

brier = ((np.array(p_pred) - np.array(y)) ** 2).mean()
print(f"\nBrier score: {brier:.4f}")

# ── 9. Save results ─────────────────────────────────────────────────────────

pr_df.to_csv("mirt_person_results.csv", index=False)

item_results = []
for j, item_id in enumerate(items):
    row = {"sample_id": item_id}
    for dim in range(D):
        row[f"a_d{dim+1}_mean"] = float(a_means[j, dim])
        row[f"a_d{dim+1}_sd"] = float(a_sds[j, dim])
    row["d_mean"] = float(d_means[j])
    row["raw_p"] = float(np.nanmean(response_matrix[:, j]))
    item_results.append(row)

pd.DataFrame(item_results).to_csv("mirt_item_results.csv", index=False)

# Save losses for convergence plot
pd.DataFrame({"step": range(len(losses)), "loss": losses}).to_csv(
    "mirt_svi_losses.csv", index=False
)

print("\nSaved: mirt_person_results.csv, mirt_item_results.csv, mirt_svi_losses.csv")
print("Done.")