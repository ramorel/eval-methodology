"""
Generalizability Study: Variance Decomposition
================================================
Decomposes total score variance into:
  - Model (fixed facet, but treated as random for variance estimation)
  - Scaffold (fixed facet)
  - Seed (random facet, nested within model × scaffold)
  - Item (random facet, crossed with all others)
  - Model × Scaffold interaction
  - Model × Item interaction
  - Scaffold × Item interaction
  - Residual (seed × item + error, confounded)
 
Uses both:
  1. Classical ANOVA-based expected mean squares (Brennan-style G-theory)
  2. Bayesian crossed random effects model in NumPyro for comparison
"""
 
import pandas as pd
import numpy as np
from itertools import product
 
# ── 1. Load and reshape ─────────────────────────────────────────────────────
df = pd.read_csv("results.csv")
print(f"Loaded {len(df)} rows")
print(f"Models: {sorted(df['model'].unique())}")
print(f"Scaffolds: {sorted(df['scaffold'].unique())}")
print(f"Seeds: {sorted(df['run_id'].unique())}")
print(f"Items: {df['sample_id'].nunique()}")
print()

# Create a person_id = model × scaffold × seed
df["person_id"] = df["model"] + "_" + df["scaffold"] + "_" + df["run_id"]
n_persons = df["person_id"].nunique()
n_items = df["sample_id"].nunique()
print(f"Person IDs (model × scaffold × seed): {n_persons}")
print(f"Items: {n_items}")
print(f"Expected cells: {n_persons * n_items} = {n_persons} × {n_items}")
print(f"Actual cells: {len(df)}")
print()


# ── 2. Descriptive overview ─────────────────────────────────────────────────
print("=" * 70)
print("CELL MEANS: Model × Scaffold")
print("=" * 70)
cell_means = df.groupby(["model", "scaffold"])["score"].agg(["mean", "std", "count"])
print(cell_means.round(3))
print()
 
# Grand mean
grand_mean = df["score"].mean()
print(f"Grand mean: {grand_mean:.3f}")
print()
 
# Marginal means
print("Model marginal means:")
print(df.groupby("model")["score"].mean().round(3))
print()
 
print("Scaffold marginal means:")
print(df.groupby("scaffold")["score"].mean().round(3))
print()

# ── 3. Seed-level variance by model and scaffold ────────────────────────────
print("=" * 70)
print("SEED VARIANCE: Score SD across 5 seeds within each model × scaffold cell")
print("=" * 70)
seed_scores = df.groupby(["model", "scaffold", "run_id"])["score"].mean()
seed_var = seed_scores.groupby(["model", "scaffold"]).agg(["mean", "std", "min", "max"])
print(seed_var.round(4))
print()


# ── 4. ANOVA-based variance decomposition ───────────────────────────────────
print("=" * 70)
print("ANOVA VARIANCE DECOMPOSITION")
print("=" * 70)
 
# Encode factors
models = sorted(df["model"].unique())
scaffolds = sorted(df["scaffold"].unique())
seeds = sorted(df["run_id"].unique())
items = sorted(df["sample_id"].unique())
 
n_m = len(models)
n_s = len(scaffolds)
n_r = len(seeds)  # seeds (replications)
n_i = len(items)
 
# Compute all marginal means needed for Type III-style decomposition
# Using the approach: variance component = (MS_effect - MS_error) / n_other
 
# Cell means at every level
mean_overall = df["score"].mean()
 
mean_model = df.groupby("model")["score"].mean()
mean_scaffold = df.groupby("scaffold")["score"].mean()
mean_item = df.groupby("sample_id")["score"].mean()
mean_seed = df.groupby("run_id")["score"].mean()
 
mean_ms = df.groupby(["model", "scaffold"])["score"].mean()
mean_mi = df.groupby(["model", "sample_id"])["score"].mean()
mean_si = df.groupby(["scaffold", "sample_id"])["score"].mean()
mean_mr = df.groupby(["model", "run_id"])["score"].mean()

# Sum of squares decomposition
# SS_model
ss_model = n_s * n_r * n_i * sum((mean_model[m] - mean_overall) ** 2 for m in models)
 
# SS_scaffold
ss_scaffold = n_m * n_r * n_i * sum(
    (mean_scaffold[s] - mean_overall) ** 2 for s in scaffolds
)
 
# SS_item
ss_item = n_m * n_s * n_r * sum(
    (mean_item[i] - mean_overall) ** 2 for i in items
)
 
# SS_model×scaffold
ss_ms = n_r * n_i * sum(
    (mean_ms[(m, s)] - mean_model[m] - mean_scaffold[s] + mean_overall) ** 2
    for m in models
    for s in scaffolds
)
 
# SS_model×item
ss_mi = n_s * n_r * sum(
    (mean_mi[(m, i)] - mean_model[m] - mean_item[i] + mean_overall) ** 2
    for m in models
    for i in items
)
 
# SS_scaffold×item
ss_si = n_m * n_r * sum(
    (mean_si[(s, i)] - mean_scaffold[s] - mean_item[i] + mean_overall) ** 2
    for s in scaffolds
    for i in items
)
 
# SS_total
ss_total = sum((row["score"] - mean_overall) ** 2 for _, row in df.iterrows())
 
# SS_residual (confounds seed, seed×item, seed×scaffold, 3-way, and error)
ss_residual = ss_total - ss_model - ss_scaffold - ss_item - ss_ms - ss_mi - ss_si

# Degrees of freedom
df_model = n_m - 1
df_scaffold = n_s - 1
df_item = n_i - 1
df_ms = (n_m - 1) * (n_s - 1)
df_mi = (n_m - 1) * (n_i - 1)
df_si = (n_s - 1) * (n_i - 1)
df_total = n_m * n_s * n_r * n_i - 1
df_residual = df_total - df_model - df_scaffold - df_item - df_ms - df_mi - df_si

# Mean squares
ms_model = ss_model / df_model
ms_scaffold = ss_scaffold / df_scaffold
ms_item = ss_item / df_item
ms_ms = ss_ms / df_ms
ms_mi = ss_mi / df_mi
ms_si = ss_si / df_si
ms_residual = ss_residual / df_residual

# Variance components (using expected mean squares)
# These are approximate — proper G-theory uses Brennan's EMS algorithm
# but for a balanced design, the formulas are straightforward
var_residual = ms_residual
var_si = (ms_si - ms_residual) / (n_m * n_r)
var_mi = (ms_mi - ms_residual) / (n_s * n_r)
var_ms = (ms_ms - ms_residual) / (n_r * n_i)
var_item = (ms_item - ms_si - ms_mi + ms_residual) / (n_m * n_s * n_r)
var_scaffold = (ms_scaffold - ms_si - ms_ms + ms_residual) / (n_m * n_r * n_i)
var_model = (ms_model - ms_mi - ms_ms + ms_residual) / (n_s * n_r * n_i)
 
# Collect and compute percentages
components = {
    "Model": var_model,
    "Scaffold": var_scaffold,
    "Item": var_item,
    "Model×Scaffold": var_ms,
    "Model×Item": var_mi,
    "Scaffold×Item": var_si,
    "Residual (seed×item+e)": var_residual,
}

# Clamp negatives to zero (can happen with small variance components)
components = {k: max(0, v) for k, v in components.items()}
total_var = sum(components.values())
 
print(f"\n{'Component':<25} {'Variance':>10} {'% of Total':>10}")
print("-" * 47)
for name, var in components.items():
    pct = 100 * var / total_var if total_var > 0 else 0
    print(f"{name:<25} {var:>10.5f} {pct:>9.1f}%")
print("-" * 47)
print(f"{'Total':<25} {total_var:>10.5f} {'100.0':>9}%")
 
print()
print("INTERPRETATION:")
print(f"  Universe-score variance (Model + Scaffold + M×S): {100*(components['Model'] + components['Scaffold'] + components['Model×Scaffold'])/total_var:.1f}%")
print(f"  Item facet (Item + interactions): {100*(components['Item'] + components['Model×Item'] + components['Scaffold×Item'])/total_var:.1f}%")
print(f"  Residual/error: {100*components['Residual (seed×item+e)']/total_var:.1f}%")


# ── 5. Item-level analysis ──────────────────────────────────────────────────
print()
print("=" * 70)
print("ITEM-LEVEL: Which items show scaffold × model interactions?")
print("=" * 70)
 
# For each item, compute the model × scaffold interaction magnitude
item_interactions = []
for item_id in items:
    item_df = df[df["sample_id"] == item_id]
    cell_means_item = item_df.groupby(["model", "scaffold"])["score"].mean()
    
    # Item-level interaction: SD of residuals after removing main effects
    item_grand = item_df["score"].mean()
    item_model_means = item_df.groupby("model")["score"].mean()
    item_scaffold_means = item_df.groupby("scaffold")["score"].mean()
    
    residuals = []
    for m in models:
        for s in scaffolds:
            if (m, s) in cell_means_item.index:
                observed = cell_means_item[(m, s)]
                expected = item_model_means[m] + item_scaffold_means[s] - item_grand
                residuals.append(observed - expected)
    
    interaction_sd = np.std(residuals) if residuals else 0
    
    # Also get the question text for interpretability
    q_text = item_df["question"].iloc[0] if "question" in item_df.columns else ""
    
    item_interactions.append({
        "sample_id": item_id,
        "question": q_text[:80],
        "overall_mean": item_grand,
        "interaction_sd": interaction_sd,
        "model_range": item_model_means.max() - item_model_means.min(),
        "scaffold_range": item_scaffold_means.max() - item_scaffold_means.min(),
    })
 
item_int_df = pd.DataFrame(item_interactions).sort_values("interaction_sd", ascending=False)
 
print("\nTop 10 items by model × scaffold interaction magnitude:")
print(item_int_df.head(10)[["sample_id", "overall_mean", "model_range", "scaffold_range", "interaction_sd", "question"]].to_string(index=False))
 
print("\n\nItems with NO variance (all correct or all incorrect):")
no_var = item_int_df[
    (item_int_df["model_range"] == 0) & 
    (item_int_df["scaffold_range"] == 0) & 
    (item_int_df["interaction_sd"] == 0)
]
print(f"  Count: {len(no_var)} of {n_items}")
if len(no_var) > 0:
    print(no_var[["sample_id", "overall_mean", "question"]].to_string(index=False))
 

# ── 6. Generalizability coefficient ─────────────────────────────────────────
print()
print("=" * 70)
print("GENERALIZABILITY COEFFICIENTS")
print("=" * 70)
 
# For a decision about models (the object of measurement),
# with scaffolds and seeds as facets:
# G = var_model / (var_model + var_mi/n_i + var_residual/(n_s*n_r*n_i))
# This tells us: how reliable is the model ranking?
 
var_m = components["Model"]
var_s = components["Scaffold"]
var_i = components["Item"]
var_ms_c = components["Model×Scaffold"]
var_mi_c = components["Model×Item"]
var_si_c = components["Scaffold×Item"]
var_e = components["Residual (seed×item+e)"]
 
# Relative error variance for model decisions (averaging over scaffolds, seeds, items)
rel_error = var_mi_c / n_i + var_ms_c / n_s + var_e / (n_s * n_r * n_i)
g_coeff = var_m / (var_m + rel_error) if (var_m + rel_error) > 0 else 0
 
print(f"  G coefficient (model ranking reliability): {g_coeff:.3f}")
print(f"  Relative error variance: {rel_error:.5f}")
print(f"  Model variance: {var_m:.5f}")
print()
 
# D-study: what if we changed the number of items or seeds?
print("D-study: G coefficient under different designs")
print(f"  {'Items':>6} {'Seeds':>6} {'Scaffolds':>10} {'G':>8}")
for ni, nr, ns in [(50, 5, 3), (50, 3, 3), (50, 1, 1), (25, 5, 3), (100, 5, 3), (50, 10, 3)]:
    rel_e = var_mi_c / ni + var_ms_c / ns + var_e / (ns * nr * ni)
    g = var_m / (var_m + rel_e) if (var_m + rel_e) > 0 else 0
    marker = " ← current" if (ni == n_i and nr == n_r and ns == n_s) else ""
    print(f"  {ni:>6} {nr:>6} {ns:>10} {g:>8.3f}{marker}")