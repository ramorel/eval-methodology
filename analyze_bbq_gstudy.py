"""
BBQ Generalizability Study: Variance Decomposition
====================================================
Decomposes variance in BBQ scores into facets:
  - Model (3 levels)
  - Persona/Scaffold (8 levels)
  - Seed (5 levels)
  - Category (9 bias categories)
  - Context condition (ambig vs disambig)
  - Item (nested within category × context)

Runs separately on:
  1. correct — accuracy (did model get the right answer?)
  2. stereotyped — bias (did model pick the stereotyped answer?)

Also computes G coefficients and D-study projections.
"""

import pandas as pd
import numpy as np
from itertools import product


def run_gstudy(df, score_col, label):
    """Run a full G-study variance decomposition on the given score column."""

    print()
    print("=" * 70)
    print(f"G-STUDY: {label} ({score_col})")
    print("=" * 70)


    # ── Descriptives ────────────────────────────────────────────────────
    grand_mean = df[score_col].mean()
    print(f"\nGrand mean: {grand_mean:.3f}")
    print(f"Total observations: {len(df)}")

    print(f"\nModel marginal means:")
    print(df.groupby("model")[score_col].mean().round(3).to_string())

    print(f"\nPersona marginal means:")
    print(df.groupby("scaffold")[score_col].mean().round(3).to_string())

    print(f"\nContext condition marginal means:")
    print(df.groupby("context_condition")[score_col].mean().round(3).to_string())

    print(f"\nCategory marginal means:")
    print(df.groupby("category")[score_col].mean().round(3).to_string())


    # ── Cell means: Model × Persona ─────────────────────────────────────
    print(f"\nModel × Persona cell means:")
    cell_means = df.groupby(["model", "scaffold"])[score_col].mean().unstack()
    print(cell_means.round(3))


    # ── Cell means: Model × Context condition ───────────────────────────
    print(f"\nModel × Context condition cell means:")
    cell_mc = df.groupby(["model", "context_condition"])[score_col].mean().unstack()
    print(cell_mc.round(3))


    # ── Seed variance by model × persona ────────────────────────────────
    print(f"\nSeed SD by Model × Persona (across 5 seeds):")
    seed_scores = df.groupby(["model", "scaffold", "run_id"])[score_col].mean()
    seed_var = seed_scores.groupby(["model", "scaffold"]).std().unstack()
    print(seed_var.round(4))


    # ── ANOVA-based variance decomposition ──────────────────────────────
    # Using a simplified model: Model × Persona × Item (collapsing seed into residual)
    # Person = Model × Persona × Seed
    # Item = sample_id
    print(f"\n{'─' * 50}")
    print(f"VARIANCE DECOMPOSITION")
    print(f"{'─' * 50}")

    # Define factors
    models = sorted(df["model"].unique())
    scaffolds = sorted(df["scaffold"].unique())
    seeds = sorted(df["run_id"].unique())
    items = sorted(df["sample_id"].unique())

    n_m = len(models)
    n_s = len(scaffolds)
    n_r = len(seeds)
    n_i = len(items)

    mean_overall = df[score_col].mean()

    # Marginal means
    mean_model = df.groupby("model")[score_col].mean()
    mean_scaffold = df.groupby("scaffold")[score_col].mean()
    mean_item = df.groupby("sample_id")[score_col].mean()

    mean_ms = df.groupby(["model", "scaffold"])[score_col].mean()
    mean_mi = df.groupby(["model", "sample_id"])[score_col].mean()
    mean_si = df.groupby(["scaffold", "sample_id"])[score_col].mean()

    # Sum of squares
    ss_model = n_s * n_r * n_i * sum(
        (mean_model[m] - mean_overall) ** 2 for m in models
    )
    ss_scaffold = n_m * n_r * n_i * sum(
        (mean_scaffold[s] - mean_overall) ** 2 for s in scaffolds
    )
    ss_item = n_m * n_s * n_r * sum(
        (mean_item[i] - mean_overall) ** 2 for i in items
    )
    ss_ms = n_r * n_i * sum(
        (mean_ms[(m, s)] - mean_model[m] - mean_scaffold[s] + mean_overall) ** 2
        for m in models for s in scaffolds
    )
    ss_mi = n_s * n_r * sum(
        (mean_mi[(m, i)] - mean_model[m] - mean_item[i] + mean_overall) ** 2
        for m in models for i in items
    )
    ss_si = n_m * n_r * sum(
        (mean_si[(s, i)] - mean_scaffold[s] - mean_item[i] + mean_overall) ** 2
        for s in scaffolds for i in items
    )
    ss_total = ((df[score_col] - mean_overall) ** 2).sum()
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

    # Variance components (EMS for balanced design)
    var_residual = ms_residual
    var_si = (ms_si - ms_residual) / (n_m * n_r)
    var_mi = (ms_mi - ms_residual) / (n_s * n_r)
    var_ms = (ms_ms - ms_residual) / (n_r * n_i)
    var_item = (ms_item - ms_si - ms_mi + ms_residual) / (n_m * n_s * n_r)
    var_scaffold = (ms_scaffold - ms_si - ms_ms + ms_residual) / (n_m * n_r * n_i)
    var_model = (ms_model - ms_mi - ms_ms + ms_residual) / (n_s * n_r * n_i)

    components = {
        "Model": var_model,
        "Persona": var_scaffold,
        "Item": var_item,
        "Model×Persona": var_ms,
        "Model×Item": var_mi,
        "Persona×Item": var_si,
        "Residual (seed×item+e)": var_residual,
    }

    # Clamp negatives
    components = {k: max(0, v) for k, v in components.items()}
    total_var = sum(components.values())

    print(f"\n{'Component':<25} {'Variance':>10} {'% of Total':>10}")
    print("-" * 47)
    for name, var in components.items():
        pct = 100 * var / total_var if total_var > 0 else 0
        print(f"{name:<25} {var:>10.5f} {pct:>9.1f}%")
    print("-" * 47)
    print(f"{'Total':<25} {total_var:>10.5f} {'100.0':>9}%")

    print(f"\nSummary:")
    obj = components["Model"] + components["Persona"] + components["Model×Persona"]
    facet = components["Item"] + components["Model×Item"] + components["Persona×Item"]
    resid = components["Residual (seed×item+e)"]
    print(f"  Object of measurement (Model+Persona+M×P): {100*obj/total_var:.1f}%")
    print(f"  Item facet (Item + interactions): {100*facet/total_var:.1f}%")
    print(f"  Residual/error: {100*resid/total_var:.1f}%")


    # ── G coefficients ──────────────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print(f"GENERALIZABILITY COEFFICIENTS")
    print(f"{'─' * 50}")

    var_m = components["Model"]
    var_p = components["Persona"]
    var_i = components["Item"]
    var_mp = components["Model×Persona"]
    var_mi_c = components["Model×Item"]
    var_pi = components["Persona×Item"]
    var_e = components["Residual (seed×item+e)"]

    # G coefficient for model ranking (averaging over personas, seeds, items)
    rel_error = var_mi_c / n_i + var_mp / n_s + var_e / (n_s * n_r * n_i)
    g_coeff = var_m / (var_m + rel_error) if (var_m + rel_error) > 0 else 0

    print(f"  G coefficient (model ranking): {g_coeff:.3f}")
    print(f"  Model variance: {var_m:.5f}")
    print(f"  Relative error: {rel_error:.5f}")

    # D-study
    print(f"\n  D-study: G under different designs")
    print(f"  {'Items':>6} {'Seeds':>6} {'Personas':>9} {'G':>8}")
    for ni, nr, ns in [
        (270, 5, 8),
        (270, 1, 1),
        (135, 5, 8),
        (50, 5, 8),
        (270, 5, 3),
        (270, 10, 8),
        (500, 5, 8),
    ]:
        rel_e = var_mi_c / ni + var_mp / ns + var_e / (ns * nr * ni)
        g = var_m / (var_m + rel_e) if (var_m + rel_e) > 0 else 0
        marker = " ← current" if (ni == n_i and nr == n_r and ns == n_s) else ""
        print(f"  {ni:>6} {nr:>6} {ns:>9} {g:>8.3f}{marker}")


    # ── Category-level analysis ─────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print(f"MODEL × CATEGORY INTERACTION")
    print(f"{'─' * 50}")

    mc_table = df.groupby(["model", "category"])[score_col].mean().unstack()
    print(f"\n{score_col} by Model × Category:")
    print(mc_table.round(3))

    # Range across models per category
    print(f"\nModel range (max - min) per category:")
    for cat in sorted(df["category"].unique()):
        cat_means = mc_table[cat]
        rng = cat_means.max() - cat_means.min()
        print(f"  {cat:<25} {rng:.3f}")

    return components


# ── Main ─────────────────────────────────────────────────────────────────────
df = pd.read_csv("bbq_results.csv")
print(f"Loaded {len(df)} rows")
print(f"Models: {sorted(df['model'].unique())}")
print(f"Personas: {sorted(df['scaffold'].unique())}")
print(f"Seeds: {sorted(df['run_id'].unique())}")
print(f"Categories: {sorted(df['category'].unique())}")
print(f"Items: {df['sample_id'].nunique()}")

# Run G-study on accuracy
components_acc = run_gstudy(df, "correct", "ACCURACY")

# Run G-study on stereotyped responding
components_bias = run_gstudy(df, "stereotyped", "STEREOTYPED RESPONDING (BIAS)")


# ── Comparison ───────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("COMPARISON: ACCURACY vs BIAS VARIANCE DECOMPOSITION")
print("=" * 70)

total_acc = sum(components_acc.values())
total_bias = sum(components_bias.values())

print(f"\n{'Component':<25} {'Accuracy %':>12} {'Bias %':>12}")
print("-" * 51)
for key in components_acc:
    pct_acc = 100 * components_acc[key] / total_acc if total_acc > 0 else 0
    pct_bias = 100 * components_bias[key] / total_bias if total_bias > 0 else 0
    print(f"{key:<25} {pct_acc:>11.1f}% {pct_bias:>11.1f}%")