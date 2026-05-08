"""
Exploratory Factor Analysis on TruthfulQA Items
================================================
Examines the latent structure of the 50-item benchmark
using the 45-person (model × scaffold × seed) response matrix.

Key questions:
  1. Is TruthfulQA unidimensional? (If not, single-score reporting is misspecified)
  2. Do empirical factors align with content categories?
  3. Do IRT discrimination parameters track factor loadings?
  4. Do scaffolds differentially affect different factors?

Uses tetrachoric correlations (appropriate for binary data)
and parallel analysis for factor retention.

Dependencies: factor_analyzer, scipy, sklearn
  pip install factor_analyzer --break-system-packages
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── 1. Load and build response matrix ───────────────────────────────────────

df = pd.read_csv("results.csv")
df["person_id"] = df["model"] + "_" + df["scaffold"] + "_" + df["run_id"]

persons = sorted(df["person_id"].unique())
items = sorted(df["sample_id"].unique())
person_to_idx = {p: i for i, p in enumerate(persons)}
item_to_idx = {it: i for i, it in enumerate(items)}

N = len(persons)
J = len(items)

# N × J response matrix
R = np.full((N, J), np.nan)
for _, row in df.iterrows():
    pi = person_to_idx[row["person_id"]]
    ji = item_to_idx[row["sample_id"]]
    R[pi, ji] = row["score"]

print(f"Response matrix: {N} persons × {J} items")
print(f"Overall mean: {np.nanmean(R):.3f}")
print()


# ── 2. Item-level descriptives ──────────────────────────────────────────────
item_means = np.nanmean(R, axis=0)
item_vars = np.nanvar(R, axis=0)

# Get category metadata
item_meta = df.drop_duplicates("sample_id").set_index("sample_id")
categories = [item_meta.loc[it, "category"] if it in item_meta.index else "" for it in items]
questions = [
    item_meta.loc[it, "question"][:70] if it in item_meta.index else "" for it in items
]

print("=" * 70)
print("ITEM DESCRIPTIVES")
print("=" * 70)
print(f"\n{'ID':>4} {'p':>5} {'var':>6} {'Category':<20} {'Question':<50}")
print("-" * 90)
for j, it in enumerate(items):
    print(
        f"{it:>4} {item_means[j]:>5.2f} {item_vars[j]:>6.3f} "
        f"{categories[j]:<20} {questions[j]:<50}"
    )

# Flag items with zero variance (can't enter correlation matrix)
zero_var_mask = item_vars == 0
n_zero_var = zero_var_mask.sum()
print(f"\nItems with zero variance: {n_zero_var} of {J}")
if n_zero_var > 0:
    print("  These items are answered identically by all 45 persons.")
    print("  They carry no information and will be excluded from factor analysis.")
    for j in np.where(zero_var_mask)[0]:
        print(f"    Item {items[j]}: p = {item_means[j]:.2f} — {questions[j]}")

# Filter to items with variance
valid_items_mask = ~zero_var_mask
R_filtered = R[:, valid_items_mask]
valid_items = [it for it, v in zip(items, valid_items_mask) if v]
valid_categories = [c for c, v in zip(categories, valid_items_mask) if v]
valid_questions = [q for q, v in zip(questions, valid_items_mask) if v]
J_valid = len(valid_items)
print(f"\nProceeding with {J_valid} items that have variance.\n")


# ── 3. Tetrachoric correlation matrix ───────────────────────────────────────
print("=" * 70)
print("COMPUTING TETRACHORIC CORRELATIONS")
print("=" * 70)


def tetrachoric_corr(x, y):
    """Compute tetrachoric correlation for two binary variables.

    Uses the approximation via Pearson's r on the raw binary data
    adjusted by the normal ogive. For a more precise estimate we
    use the maximum-likelihood approach when scipy is available,
    falling back to the cosine-pi approximation (Bonett & Price, 2005).
    """
    # Build 2×2 table
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

    # Add 0.5 continuity correction if any cell is zero
    if a * d == 0 or b * c == 0:
        a += 0.5
        b += 0.5
        c += 0.5
        d += 0.5

    # Cosine-pi approximation: r_tet ≈ cos(π / (1 + sqrt(ad/bc)))
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

# Check for positive semi-definiteness and fix if needed
eigenvalues = np.linalg.eigvalsh(tet_corr)
if np.any(eigenvalues < 0):
    print(f"  Correlation matrix has {(eigenvalues < 0).sum()} negative eigenvalues.")
    print("  Applying nearest PSD correction...")
    # Nearest PSD: clip eigenvalues at a small positive value
    eigvals, eigvecs = np.linalg.eigh(tet_corr)
    eigvals = np.maximum(eigvals, 1e-6)
    tet_corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Re-normalize to unit diagonal
    d = np.sqrt(np.diag(tet_corr))
    tet_corr = tet_corr / np.outer(d, d)
    np.fill_diagonal(tet_corr, 1.0)

print(f"  Tetrachoric correlation matrix: {J_valid} × {J_valid}")
print(f"  Mean off-diagonal r: {tet_corr[np.triu_indices(J_valid, k=1)].mean():.3f}")
print(f"  Range: [{tet_corr[np.triu_indices(J_valid, k=1)].min():.3f}, "
      f"{tet_corr[np.triu_indices(J_valid, k=1)].max():.3f}]")
print()


# ── 4. Parallel analysis for factor retention ──────────────────────────────
print("=" * 70)
print("PARALLEL ANALYSIS — How many factors?")
print("=" * 70)

# Eigenvalues of observed tetrachoric correlation matrix
obs_eigenvalues = np.sort(np.linalg.eigvalsh(tet_corr))[::-1]

# Generate random data eigenvalues (parallel analysis)
n_simulations = 200
random_eigenvalues = np.zeros((n_simulations, J_valid))
successful_sims = 0
attempts = 0
max_attempts = n_simulations * 5
 
while successful_sims < n_simulations and attempts < max_attempts:
    attempts += 1
    # Generate random binary data with same marginal proportions
    R_random = np.zeros_like(R_filtered)
    for j in range(J_valid):
        p = np.nanmean(R_filtered[:, j])
        # Clamp p away from 0/1 to avoid zero-variance columns
        p = np.clip(p, 0.05, 0.95)
        R_random[:, j] = np.random.binomial(1, p, size=N)
 
    # Skip if any column has zero variance
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
 
print(f"  Completed {successful_sims} of {n_simulations} simulations ({attempts} attempts)")
if successful_sims < n_simulations:
    random_eigenvalues = random_eigenvalues[:successful_sims]
 
# 95th percentile of random eigenvalues
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
 
# Also check scree — ratio of successive eigenvalues
print("\nScree ratios (eigenvalue[k] / eigenvalue[k+1]):")
for k in range(min(8, J_valid - 1)):
    ratio = obs_eigenvalues[k] / obs_eigenvalues[k + 1] if obs_eigenvalues[k + 1] > 0 else float("inf")
    print(f"  {k+1}→{k+2}: {ratio:.2f}")


# ── 5. Factor analysis ─────────────────────────────────────────────────────
print()
print("=" * 70)
print("EXPLORATORY FACTOR ANALYSIS")
print("=" * 70)

try:
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import calculate_kmo

    # KMO test for factorability
    try:
        kmo_all, kmo_model = calculate_kmo(R_filtered)
        print(f"\nKMO measure of sampling adequacy: {kmo_model:.3f}")
        if kmo_model < 0.5:
            print("  WARNING: KMO < 0.5 suggests factor analysis may not be appropriate.")
            print("  (This is expected with only 45 persons and binary data.)")
    except Exception:
        print("\nKMO computation failed (common with binary data). Proceeding anyway.")

    # Determine number of factors to extract
    n_factors = max(1, min(n_factors_parallel, 5))  # Cap at 5
    print(f"\nExtracting {n_factors} factor(s) with oblimin rotation...")

    # Fit on the tetrachoric correlation matrix
    fa = FactorAnalyzer(
        n_factors=n_factors,
        rotation="oblimin" if n_factors > 1 else None,
        method="minres",
        is_corr_matrix=True,
    )
    fa.fit(tet_corr)

    # Loadings
    loadings = fa.loadings_
    communalities = fa.get_communalities()

    print(f"\nFactor loadings (oblimin rotation):")
    print(f"{'ID':>4} {'h²':>5}", end="")
    for f in range(n_factors):
        print(f" {'F' + str(f+1):>7}", end="")
    print(f"  {'Category':<18} {'Question':<45}")
    print("-" * (35 + n_factors * 8 + 65))

    # Sort by highest absolute loading on first factor
    sort_idx = np.argsort(-np.abs(loadings[:, 0]))
    for idx in sort_idx:
        print(f"{valid_items[idx]:>4} {communalities[idx]:>5.2f}", end="")
        for f in range(n_factors):
            val = loadings[idx, f]
            marker = "*" if abs(val) > 0.3 else " "
            print(f" {val:>6.2f}{marker}", end="")
        print(f"  {valid_categories[idx]:<18} {valid_questions[idx]:<45}")

    # Factor correlations (if multi-factor)
    if n_factors > 1:
        print("\nFactor correlation matrix:")
        factor_corr = fa.phi_ if hasattr(fa, "phi_") and fa.phi_ is not None else np.eye(n_factors)
        for i in range(n_factors):
            row = "  ".join(f"{factor_corr[i, j]:>6.3f}" for j in range(n_factors))
            print(f"  F{i+1}: {row}")

    # Variance explained
    ev = fa.get_factor_variance()
    print(f"\nVariance explained:")
    print(f"  {'Factor':>8} {'SS Loading':>11} {'Proportion':>11} {'Cumulative':>11}")
    for f in range(n_factors):
        print(f"  {'F' + str(f+1):>8} {ev[0][f]:>11.3f} {ev[1][f]:>11.3f} {ev[2][f]:>11.3f}")


    # ── 6. Map factors to content categories ────────────────────────────────
    print()
    print("=" * 70)
    print("FACTOR × CATEGORY MAPPING")
    print("=" * 70)

    # For each category, compute mean absolute loading on each factor
    cat_factor_map = []
    unique_cats = sorted(set(valid_categories))
    for cat in unique_cats:
        cat_mask = [c == cat for c in valid_categories]
        cat_loadings = loadings[cat_mask]
        if len(cat_loadings) > 0:
            row = {"category": cat, "n_items": len(cat_loadings)}
            for f in range(n_factors):
                row[f"F{f+1}_mean_loading"] = cat_loadings[:, f].mean()
                row[f"F{f+1}_abs_mean"] = np.abs(cat_loadings[:, f]).mean()
            cat_factor_map.append(row)

    cat_df = pd.DataFrame(cat_factor_map)
    print(f"\nMean factor loadings by content category:")
    print(cat_df.to_string(index=False))


    # ── 7. Compare factor loadings to IRT discrimination ────────────────────
    print()
    print("=" * 70)
    print("FACTOR LOADINGS vs IRT DISCRIMINATION")
    print("=" * 70)

    try:
        irt_items = pd.read_csv("item_irt_results.csv")
        # Merge on sample_id
        comparison = []
        for j, it in enumerate(valid_items):
            irt_row = irt_items[irt_items["sample_id"] == it]
            a_mean = irt_row["a_mean"].values[0] if len(irt_row) > 0 else np.nan
            comparison.append({
                "sample_id": it,
                "irt_a": a_mean,
                "F1_loading": loadings[j, 0],
                "F1_abs": abs(loadings[j, 0]),
                "communality": communalities[j],
                "category": valid_categories[j],
            })

        comp_df = pd.DataFrame(comparison)

        if comp_df["irt_a"].notna().sum() > 5:
            r_a_f1 = comp_df[["irt_a", "F1_abs"]].corr().iloc[0, 1]
            r_a_h2 = comp_df[["irt_a", "communality"]].corr().iloc[0, 1]
            print(f"\nCorrelation between IRT discrimination (a) and |F1 loading|: {r_a_f1:.3f}")
            print(f"Correlation between IRT discrimination (a) and communality: {r_a_h2:.3f}")
            print("\n(These should be positive — items that discriminate well in IRT")
            print(" should also load strongly on the dominant factor.)")

            # Show the joint distribution
            print(f"\n{'ID':>4} {'IRT a':>7} {'|F1|':>6} {'h²':>5} {'Category':<18}")
            print("-" * 45)
            for _, row in comp_df.sort_values("irt_a", ascending=False).iterrows():
                print(
                    f"{row['sample_id']:>4} {row['irt_a']:>7.2f} "
                    f"{row['F1_abs']:>6.2f} {row['communality']:>5.2f} "
                    f"{row['category']:<18}"
                )
        else:
            print("\nIRT results not found. Run fit_irt_45.py first to enable this comparison.")

    except FileNotFoundError:
        print("\nitem_irt_results.csv not found. Run fit_irt_45.py first.")
        print("Then re-run this script to see the IRT × factor loading comparison.")

except ImportError:
    print("\nfactor_analyzer not installed. Install with:")
    print("  pip install factor_analyzer --break-system-packages")
    print("\nFalling back to eigenvalue-only analysis (already computed above).")


# ── 8. Scaffold effects by factor ───────────────────────────────────────────
print()
print("=" * 70)
print("SCAFFOLD EFFECTS BY FACTOR (Factor Score Proxies)")
print("=" * 70)

try:
    # Compute factor score proxies using regression method
    # Factor scores = R_filtered @ inv(tet_corr) @ loadings
    # (Thurstone's regression scores)
    inv_corr = np.linalg.pinv(tet_corr)
    factor_scores = R_filtered @ inv_corr @ loadings  # N × n_factors

    # Attach metadata
    fs_df = pd.DataFrame(factor_scores, columns=[f"F{f+1}" for f in range(n_factors)])
    fs_df["person_id"] = persons

    # Parse person_id back into model, scaffold, seed
    parsed = fs_df["person_id"].str.rsplit("_", n=2, expand=True)
    fs_df["model"] = parsed[0]
    fs_df["scaffold"] = parsed[1]
    fs_df["seed"] = parsed[2]

    for f in range(n_factors):
        fname = f"F{f+1}"
        print(f"\n{fname} scores by Model × Scaffold:")
        pivot = fs_df.pivot_table(values=fname, index="model", columns="scaffold", aggfunc="mean")
        print(pivot.round(3))

        # Scaffold main effect on this factor
        scaffold_means = fs_df.groupby("scaffold")[fname].mean()
        scaffold_range = scaffold_means.max() - scaffold_means.min()
        print(f"  Scaffold range on {fname}: {scaffold_range:.3f}")

        # Model main effect on this factor
        model_means = fs_df.groupby("model")[fname].mean()
        model_range = model_means.max() - model_means.min()
        print(f"  Model range on {fname}: {model_range:.3f}")

except Exception as e:
    print(f"\nCould not compute factor scores: {e}")
    print("This section requires the factor analysis to have succeeded.")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)