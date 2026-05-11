"""
Exploratory Factor Analysis on BBQ Bias Items
===============================================
Tests whether "social bias" as measured by BBQ is unidimensional
or separates into distinct latent factors (e.g., racial bias vs.
gender bias vs. age bias).

Uses the 75 informative items from the stereotyped response matrix
(120 persons × 75 items with variance).

Key questions:
  1. Is bias one thing or many things?
  2. Do empirical factors align with BBQ's bias categories?
  3. Do disambiguated vs ambiguous items load differently?
  4. How do factor loadings relate to IRT discrimination?
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── 1. Load and build response matrix ───────────────────────────────────────
df = pd.read_csv("bbq_results.csv")
df["person_id"] = df["model"] + "|" + df["scaffold"] + "|" + df["run_id"]

persons = sorted(df["person_id"].unique())
items = sorted(df["sample_id"].unique())
person_to_idx = {p: i for i, p in enumerate(persons)}
item_to_idx = {it: i for i, it in enumerate(items)}

N = len(persons)
J = len(items)

# Build stereotyped response matrix
R = np.full((N, J), np.nan)
for _, row in df.iterrows():
    pi = person_to_idx[row["person_id"]]
    ji = item_to_idx[row["sample_id"]]
    R[pi, ji] = row["stereotyped"]

print(f"Response matrix: {N} persons × {J} items")
print(f"Overall stereotyped rate: {np.nanmean(R):.3f}")
print()


# ── 2. Item descriptives and filtering ──────────────────────────────────────
item_means = np.nanmean(R, axis=0)
item_vars = np.nanvar(R, axis=0)

# Get metadata
item_meta = df.drop_duplicates("sample_id").set_index("sample_id")
categories = [
    item_meta.loc[it, "category"] if it in item_meta.index else ""
    for it in items
]
conditions = [
    item_meta.loc[it, "context_condition"] if it in item_meta.index else ""
    for it in items
]

# Filter zero-variance items
zero_var_mask = item_vars == 0
n_zero = zero_var_mask.sum()
print(f"Items with zero variance: {n_zero} of {J}")

valid_mask = ~zero_var_mask
R_filtered = R[:, valid_mask]
valid_items = [it for it, v in zip(items, valid_mask) if v]
valid_categories = [c for c, v in zip(categories, valid_mask) if v]
valid_conditions = [c for c, v in zip(conditions, valid_mask) if v]
J_valid = len(valid_items)
print(f"Proceeding with {J_valid} items that have variance.")

# Summary of informative items by category × condition
print(f"\nInformative items by category × condition:")
from collections import Counter
info_counts = Counter(zip(valid_categories, valid_conditions))
for (cat, cond), n in sorted(info_counts.items()):
    print(f"  {cat:<25} {cond:<10} {n:>3}")

print()


# ── 3. Tetrachoric correlation matrix ───────────────────────────────────────
print("=" * 70)
print("COMPUTING TETRACHORIC CORRELATIONS")
print("=" * 70)


def tetrachoric_corr(x, y):
    """Tetrachoric correlation via cosine-pi approximation."""
    both_valid = ~(np.isnan(x) | np.isnan(y))
    x_v = x[both_valid].astype(int)
    y_v = y[both_valid].astype(int)

    a = np.sum((x_v == 1) & (y_v == 1))
    b = np.sum((x_v == 1) & (y_v == 0))
    c = np.sum((x_v == 0) & (y_v == 1))
    d = np.sum((x_v == 0) & (y_v == 0))

    n = a + b + c + d
    if n == 0:
        return 0.0

    if a * d == 0 or b * c == 0:
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5

    odds_ratio = (a * d) / (b * c)
    r_tet = np.cos(np.pi / (1.0 + np.sqrt(odds_ratio)))
    return np.clip(r_tet, -1, 1)


print("Computing pairwise tetrachoric correlations...")
tet_corr = np.eye(J_valid)
for i in range(J_valid):
    for j in range(i + 1, J_valid):
        r = tetrachoric_corr(R_filtered[:, i], R_filtered[:, j])
        tet_corr[i, j] = r
        tet_corr[j, i] = r

# PSD correction
eigenvalues = np.linalg.eigvalsh(tet_corr)
if np.any(eigenvalues < 0):
    print(f"  {(eigenvalues < 0).sum()} negative eigenvalues — applying PSD correction")
    eigvals, eigvecs = np.linalg.eigh(tet_corr)
    eigvals = np.maximum(eigvals, 1e-6)
    tet_corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.sqrt(np.diag(tet_corr))
    tet_corr = tet_corr / np.outer(d, d)
    np.fill_diagonal(tet_corr, 1.0)

off_diag = tet_corr[np.triu_indices(J_valid, k=1)]
print(f"  Mean off-diagonal r: {off_diag.mean():.3f}")
print(f"  Range: [{off_diag.min():.3f}, {off_diag.max():.3f}]")
print()


# ── 4. Parallel analysis ────────────────────────────────────────────────────
print("=" * 70)
print("PARALLEL ANALYSIS — How many factors?")
print("=" * 70)

obs_eigenvalues = np.sort(np.linalg.eigvalsh(tet_corr))[::-1]

n_simulations = 200
random_eigenvalues = np.zeros((n_simulations, J_valid))
successful_sims = 0
attempts = 0
max_attempts = n_simulations * 5

while successful_sims < n_simulations and attempts < max_attempts:
    attempts += 1
    R_random = np.zeros_like(R_filtered)
    for j in range(J_valid):
        p = np.clip(np.nanmean(R_filtered[:, j]), 0.05, 0.95)
        R_random[:, j] = np.random.binomial(1, p, size=N)

    if np.any(R_random.var(axis=0) == 0):
        continue

    try:
        rand_corr = np.corrcoef(R_random.T)
        random_eigenvalues[successful_sims] = np.sort(
            np.linalg.eigvalsh(rand_corr)
        )[::-1]
        successful_sims += 1
    except np.linalg.LinAlgError:
        continue

print(f"  Completed {successful_sims} simulations ({attempts} attempts)")
if successful_sims < n_simulations:
    random_eigenvalues = random_eigenvalues[:successful_sims]

threshold = np.percentile(random_eigenvalues, 95, axis=0)

print(f"\n{'Factor':>7} {'Observed':>10} {'95th %ile':>10} {'Retain?':>8}")
print("-" * 38)
n_factors_parallel = 0
for k in range(min(15, J_valid)):
    retain = "  YES" if obs_eigenvalues[k] > threshold[k] else "  no"
    if obs_eigenvalues[k] > threshold[k]:
        n_factors_parallel = k + 1
    print(f"{k+1:>7} {obs_eigenvalues[k]:>10.3f} {threshold[k]:>10.3f} {retain:>8}")

print(f"\nParallel analysis suggests: {n_factors_parallel} factor(s)")

# Scree ratios
print("\nScree ratios:")
for k in range(min(8, J_valid - 1)):
    ratio = (
        obs_eigenvalues[k] / obs_eigenvalues[k + 1]
        if obs_eigenvalues[k + 1] > 0
        else float("inf")
    )
    print(f"  {k+1}→{k+2}: {ratio:.2f}")


# ── 5. Factor analysis ─────────────────────────────────────────────────────
print()
print("=" * 70)
print("EXPLORATORY FACTOR ANALYSIS")
print("=" * 70)

try:
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import calculate_kmo

    try:
        kmo_all, kmo_model = calculate_kmo(R_filtered)
        print(f"\nKMO: {kmo_model:.3f}")
    except Exception:
        print("\nKMO computation failed. Proceeding.")

    n_factors = max(1, min(n_factors_parallel, 9))  # cap at 9 (num categories)
    print(f"Extracting {n_factors} factor(s) with oblimin rotation...")

    fa = FactorAnalyzer(
        n_factors=n_factors,
        rotation="oblimin" if n_factors > 1 else None,
        method="minres",
        is_corr_matrix=True,
    )
    fa.fit(tet_corr)

    loadings = fa.loadings_
    communalities = fa.get_communalities()

    # Print loadings sorted by highest loading
    print(f"\nFactor loadings:")
    print(f"{'ID':>4} {'h²':>5}", end="")
    for f in range(n_factors):
        print(f" {'F'+str(f+1):>7}", end="")
    print(f"  {'p':>5} {'Category':<22} {'Condition':<10}")
    print("-" * (12 + n_factors * 8 + 5 + 22 + 10))

    # Sort by primary loading
    primary_factor = np.argmax(np.abs(loadings), axis=1)
    sort_idx = np.lexsort((-np.max(np.abs(loadings), axis=1), primary_factor))

    for idx in sort_idx:
        item_id = valid_items[idx]
        p = np.nanmean(R_filtered[:, idx])
        print(f"{item_id:>4} {communalities[idx]:>5.2f}", end="")
        for f in range(n_factors):
            val = loadings[idx, f]
            marker = "*" if abs(val) > 0.3 else " "
            print(f" {val:>6.2f}{marker}", end="")
        print(
            f"  {p:>5.2f} {valid_categories[idx]:<22} {valid_conditions[idx]:<10}"
        )

    # Factor correlations
    if n_factors > 1:
        print("\nFactor correlation matrix:")
        factor_corr = (
            fa.phi_
            if hasattr(fa, "phi_") and fa.phi_ is not None
            else np.eye(n_factors)
        )
        for i in range(n_factors):
            row = "  ".join(f"{factor_corr[i, j]:>6.3f}" for j in range(n_factors))
            print(f"  F{i+1}: {row}")

    # Variance explained
    ev = fa.get_factor_variance()
    print(f"\nVariance explained:")
    print(f"  {'Factor':>8} {'SS Loading':>11} {'Proportion':>11} {'Cumulative':>11}")
    for f in range(n_factors):
        print(
            f"  {'F'+str(f+1):>8} {ev[0][f]:>11.3f} "
            f"{ev[1][f]:>11.3f} {ev[2][f]:>11.3f}"
        )


    # ── 6. Factor × Category mapping ────────────────────────────────────
    print()
    print("=" * 70)
    print("FACTOR × BIAS CATEGORY MAPPING")
    print("=" * 70)
    print("\nThis is the key test: do factors align with bias categories?")

    unique_cats = sorted(set(valid_categories))
    cat_rows = []
    for cat in unique_cats:
        cat_mask = [c == cat for c in valid_categories]
        cat_loadings = loadings[cat_mask]
        if len(cat_loadings) > 0:
            row = {"category": cat, "n_items": len(cat_loadings)}
            for f in range(n_factors):
                row[f"F{f+1}_mean"] = cat_loadings[:, f].mean()
                row[f"F{f+1}_abs"] = np.abs(cat_loadings[:, f]).mean()
            cat_rows.append(row)

    cat_df = pd.DataFrame(cat_rows)
    print(cat_df.to_string(index=False))


    # ── 7. Factor × Context condition ───────────────────────────────────
    print()
    print("=" * 70)
    print("FACTOR × CONTEXT CONDITION")
    print("=" * 70)

    for cond in ["ambig", "disambig"]:
        cond_mask = [c == cond for c in valid_conditions]
        cond_loadings = loadings[cond_mask]
        if len(cond_loadings) > 0:
            print(f"\n{cond} items ({sum(cond_mask)}):")
            for f in range(n_factors):
                mean_load = cond_loadings[:, f].mean()
                abs_load = np.abs(cond_loadings[:, f]).mean()
                print(f"  F{f+1}: mean={mean_load:.3f}, |mean|={abs_load:.3f}")


    # ── 8. Compare to IRT discrimination ────────────────────────────────
    print()
    print("=" * 70)
    print("FACTOR LOADINGS vs IRT DISCRIMINATION")
    print("=" * 70)

    try:
        irt_items = pd.read_csv("bbq_irt_item_bias.csv")
        comparison = []
        for j, it in enumerate(valid_items):
            irt_row = irt_items[irt_items["sample_id"] == it]
            a_mean = irt_row["a_mean"].values[0] if len(irt_row) > 0 else np.nan
            comparison.append({
                "sample_id": it,
                "irt_a": a_mean,
                "max_loading": np.max(np.abs(loadings[j])),
                "communality": communalities[j],
                "category": valid_categories[j],
                "condition": valid_conditions[j],
            })

        comp_df = pd.DataFrame(comparison)
        if comp_df["irt_a"].notna().sum() > 5:
            r_a_load = comp_df[["irt_a", "max_loading"]].corr().iloc[0, 1]
            r_a_h2 = comp_df[["irt_a", "communality"]].corr().iloc[0, 1]
            print(f"\nCorrelation IRT discrimination vs |max loading|: {r_a_load:.3f}")
            print(f"Correlation IRT discrimination vs communality: {r_a_h2:.3f}")
    except FileNotFoundError:
        print("\nbbq_irt_item_bias.csv not found. Run fit_bbq_irt.py first.")


    # ── 9. Dominant factor per item — is bias category predicted? ────────
    print()
    print("=" * 70)
    print("DOES FACTOR ASSIGNMENT PREDICT BIAS CATEGORY?")
    print("=" * 70)

    if n_factors > 1:
        # Assign each item to its dominant factor
        dominant_factor = np.argmax(np.abs(loadings), axis=1)

        # Cross-tabulate
        crosstab = pd.DataFrame({
            "category": valid_categories,
            "condition": valid_conditions,
            "dominant_factor": [f"F{f+1}" for f in dominant_factor],
        })

        print("\nCross-tabulation: Category × Dominant Factor")
        ct = pd.crosstab(crosstab["category"], crosstab["dominant_factor"])
        print(ct)

        print("\nCross-tabulation: Condition × Dominant Factor")
        ct2 = pd.crosstab(crosstab["condition"], crosstab["dominant_factor"])
        print(ct2)

        # Adjusted Rand Index or Cramer's V
        from collections import Counter
        total = len(valid_categories)
        n_cats = len(set(valid_categories))
        n_facs = n_factors
        # Cramer's V
        ct_array = pd.crosstab(
            crosstab["category"], crosstab["dominant_factor"]
        ).values
        from scipy.stats import chi2_contingency
        try:
            chi2, p_val, dof, expected = chi2_contingency(ct_array)
            n_obs = ct_array.sum()
            k = min(ct_array.shape) - 1
            cramers_v = np.sqrt(chi2 / (n_obs * k)) if k > 0 else 0
            print(f"\nCramer's V (category × factor): {cramers_v:.3f} (p={p_val:.4f})")
            print(
                "  V > 0.3 suggests factors partially align with categories"
            )
            print(
                "  V < 0.1 suggests factors cut across categories"
            )
        except Exception:
            pass

except ImportError:
    print("\nfactor_analyzer not installed.")
    print("  pip install factor_analyzer --break-system-packages")

print()
print("=" * 70)
print("DONE")
print("=" * 70)