"""
2PL Bayesian IRT for BBQ
=========================
Fits separate IRT models on:
  1. correct — accuracy (ability to get the right answer)
  2. stereotyped — bias (tendency to pick the stereotyped answer)

120 persons (3 models × 8 personas × 5 seeds) × 270 items.
Items with zero variance are excluded automatically.

The bias IRT is the novel analysis: high θ = high stereotyped responding.
Item discrimination tells you which items actually probe bias vs. noise.
"""

import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import arviz as az


def fit_irt_2pl(response_matrix, persons, items, label, n_warmup=1000, n_samples=2000):
    """Fit a 2PL IRT model and return results."""

    N, J = response_matrix.shape
    print(f"\n{'=' * 70}")
    print(f"2PL IRT: {label}")
    print(f"{'=' * 70}")
    print(f"Persons: {N}, Items: {J}")

    # Filter zero-variance items
    item_vars = np.nanvar(response_matrix, axis=0)
    informative_mask = item_vars > 0
    n_informative = informative_mask.sum()
    n_dropped = J - n_informative
    print(f"Items with variance: {n_informative} of {J} ({n_dropped} dropped)")

    if n_informative < 5:
        print("  Too few informative items. Skipping.")
        return None, None

    response_filtered = response_matrix[:, informative_mask]
    items_filtered = [it for it, m in zip(items, informative_mask) if m]
    J_f = len(items_filtered)

    # Build flat observation arrays
    person_idx, item_idx_flat = np.where(~np.isnan(response_filtered))
    y = response_filtered[person_idx, item_idx_flat].astype(int)

    person_idx = jnp.array(person_idx)
    item_idx = jnp.array(item_idx_flat)
    y = jnp.array(y)
    print(f"Observations: {len(y)}")

    # Model
    def irt_2pl(person_idx, item_idx, y=None, N=None, J=None):
        theta = numpyro.sample("theta", dist.Normal(0, 1).expand([N]))
        b = numpyro.sample("b", dist.Normal(0, 2).expand([J]))
        log_a = numpyro.sample("log_a", dist.Normal(0, 0.5).expand([J]))
        a = numpyro.deterministic("a", jnp.exp(log_a))
        logit_p = a[item_idx] * (theta[person_idx] - b[item_idx])
        numpyro.sample("obs", dist.Bernoulli(logits=logit_p), obs=y)

    # Fit
    print(f"\nFitting (this may take a few minutes)...")
    kernel = NUTS(irt_2pl)
    mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples,
                num_chains=2, progress_bar=True)
    mcmc.run(jax.random.PRNGKey(42), person_idx=person_idx,
             item_idx=item_idx, y=y, N=N, J=J_f)

    # Diagnostics
    idata = az.from_numpyro(mcmc)
    samples = mcmc.get_samples()

    n_div = idata.sample_stats["diverging"].sum().item()
    theta_summary = az.summary(idata, var_names=["theta"], round_to=3)
    a_summary = az.summary(idata, var_names=["a"], round_to=3)

    print(f"\nDivergences: {n_div}")
    print(f"Max r_hat — theta: {theta_summary['r_hat'].max():.3f}, "
          f"a: {a_summary['r_hat'].max():.3f}")
    print(f"Min ESS — theta: {theta_summary['ess_bulk'].min():.0f}, "
          f"a: {a_summary['ess_bulk'].min():.0f}")

    # Extract posteriors
    theta_means = np.array(samples["theta"].mean(axis=0))
    theta_sds = np.array(samples["theta"].std(axis=0))
    a_means = np.array(samples["a"].mean(axis=0))
    a_sds = np.array(samples["a"].std(axis=0))
    b_means = np.array(samples["b"].mean(axis=0))

    # Person results
    person_results = []
    for person_id, idx in zip(persons, range(N)):
        parts = person_id.split("|")
        person_results.append({
            "person_id": person_id,
            "model": parts[0],
            "scaffold": parts[1],
            "seed": parts[2],
            "theta_mean": float(theta_means[idx]),
            "theta_sd": float(theta_sds[idx]),
            "raw_score": float(np.nanmean(response_filtered[idx])),
        })
    pr_df = pd.DataFrame(person_results)

    # Ability by model × persona
    print(f"\nθ means by Model × Persona:")
    pivot = pr_df.pivot_table(values="theta_mean", index="model",
                               columns="scaffold", aggfunc="mean")
    print(pivot.round(3))

    print(f"\nModel marginal θ:")
    print(pr_df.groupby("model")["theta_mean"].agg(["mean", "std"]).round(3))

    print(f"\nPersona marginal θ:")
    print(pr_df.groupby("scaffold")["theta_mean"].agg(["mean", "std"]).round(3))

    # Item results
    item_meta = df.drop_duplicates("sample_id").set_index("sample_id")
    item_results = []
    for j, item_id in enumerate(items_filtered):
        cat = item_meta.loc[item_id, "category"] if item_id in item_meta.index else ""
        cond = item_meta.loc[item_id, "context_condition"] if item_id in item_meta.index else ""
        q = item_meta.loc[item_id, "question_text"] if "question_text" in item_meta.columns and item_id in item_meta.index else ""
        item_results.append({
            "sample_id": item_id,
            "category": cat,
            "context_condition": cond,
            "a_mean": float(a_means[j]),
            "a_sd": float(a_sds[j]),
            "b_mean": float(b_means[j]),
            "raw_p": float(np.nanmean(response_filtered[:, j])),
        })
    ir_df = pd.DataFrame(item_results).sort_values("a_mean", ascending=False)

    print(f"\nTop 15 most discriminating items:")
    cols = ["sample_id", "a_mean", "b_mean", "raw_p", "category", "context_condition"]
    print(ir_df.head(15)[cols].to_string(index=False))

    print(f"\nBottom 10 least discriminating items:")
    print(ir_df.tail(10)[cols].to_string(index=False))

    n_high = (ir_df["a_mean"] > 1.3).sum()
    n_low = (ir_df["a_mean"] < 0.8).sum()
    print(f"\nHighly discriminating (a > 1.3): {n_high}")
    print(f"Low discrimination (a < 0.8): {n_low}")
    print(f"Near prior (0.8–1.3): {len(ir_df) - n_high - n_low}")

    # Discrimination by category
    print(f"\nMean discrimination by category:")
    cat_disc = ir_df.groupby("category")["a_mean"].agg(["mean", "std", "count"])
    print(cat_disc.round(3))

    # Discrimination by context condition
    print(f"\nMean discrimination by context condition:")
    cond_disc = ir_df.groupby("context_condition")["a_mean"].agg(["mean", "std", "count"])
    print(cond_disc.round(3))

    return pr_df, ir_df


# ── Main ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("bbq_results.csv")
df["person_id"] = df["model"] + "|" + df["scaffold"] + "|" + df["run_id"]

persons = sorted(df["person_id"].unique())
items = sorted(df["sample_id"].unique())
person_to_idx = {p: i for i, p in enumerate(persons)}
item_to_idx = {it: i for i, it in enumerate(items)}

N = len(persons)
J = len(items)

print(f"Persons: {N}, Items: {J}")


# ── Build response matrices ─────────────────────────────────────────────────
# Accuracy matrix
acc_matrix = np.full((N, J), np.nan)
for _, row in df.iterrows():
    pi = person_to_idx[row["person_id"]]
    ji = item_to_idx[row["sample_id"]]
    acc_matrix[pi, ji] = row["correct"]

# Bias (stereotyped) matrix
bias_matrix = np.full((N, J), np.nan)
for _, row in df.iterrows():
    pi = person_to_idx[row["person_id"]]
    ji = item_to_idx[row["sample_id"]]
    bias_matrix[pi, ji] = row["stereotyped"]


# ── Fit IRT on accuracy ─────────────────────────────────────────────────────
pr_acc, ir_acc = fit_irt_2pl(acc_matrix, persons, items, "ACCURACY (correct)")
if pr_acc is not None:
    pr_acc.to_csv("bbq_irt_person_accuracy.csv", index=False)
    ir_acc.to_csv("bbq_irt_item_accuracy.csv", index=False)
    print("\nSaved: bbq_irt_person_accuracy.csv, bbq_irt_item_accuracy.csv")


# ── Fit IRT on stereotyped responding ────────────────────────────────────────
pr_bias, ir_bias = fit_irt_2pl(bias_matrix, persons, items, "BIAS (stereotyped)")
if pr_bias is not None:
    pr_bias.to_csv("bbq_irt_person_bias.csv", index=False)
    ir_bias.to_csv("bbq_irt_item_bias.csv", index=False)
    print("\nSaved: bbq_irt_person_bias.csv, bbq_irt_item_bias.csv")


# ── Compare the two models ──────────────────────────────────────────────────
if pr_acc is not None and pr_bias is not None:
    print()
    print("=" * 70)
    print("COMPARISON: ACCURACY θ vs BIAS θ")
    print("=" * 70)

    merged = pr_acc[["person_id", "model", "scaffold", "theta_mean"]].rename(
        columns={"theta_mean": "theta_accuracy"}
    ).merge(
        pr_bias[["person_id", "theta_mean"]].rename(
            columns={"theta_mean": "theta_bias"}
        ),
        on="person_id",
    )

    print(f"\nCorrelation between accuracy θ and bias θ: "
          f"{merged['theta_accuracy'].corr(merged['theta_bias']):.3f}")

    print(f"\nBy model:")
    for model in sorted(merged["model"].unique()):
        m = merged[merged["model"] == model]
        print(f"  {model}: acc_θ={m['theta_accuracy'].mean():.3f}, "
              f"bias_θ={m['theta_bias'].mean():.3f}")

    print(f"\nBy persona:")
    for scaffold in sorted(merged["scaffold"].unique()):
        s = merged[merged["scaffold"] == scaffold]
        print(f"  {scaffold:<25} acc_θ={s['theta_accuracy'].mean():.3f}, "
              f"bias_θ={s['theta_bias'].mean():.3f}")