"""Iterative pairwise feature generator.

For every unordered pair of input features and every operation
(mul, div, add, sub, rel_diff = (a-b)/(a+b)) a candidate feature is built.
Candidates are screened on TRAIN data only:
  1. discard when |corr(candidate, target)| < min_target_corr;
  2. rank survivors by |corr| and greedily keep those whose |corr| with every
     already-kept candidate is below max_inter_corr, up to max_new_per_iter.
The next iteration pairs kept candidates with the whole pool (base + kept).

Kept features are described by JSON-serializable recipes so they can be
reproduced at inference time with apply_recipes().
"""
import numpy as np
import pandas as pd

OPS = {
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "rel": lambda a, b: (a - b) / (a + b),
}

_CHUNK = 512          # candidate columns per correlation batch
_MIN_VALID_ROWS = 100  # candidates with fewer finite train rows are discarded


def apply_recipes(df: pd.DataFrame, recipes: list[dict]) -> pd.DataFrame:
    """Append generated features to df. Recipes are ordered, so a recipe may
    reference the output of an earlier one."""
    new_cols = {}
    for r in recipes:
        # Skip recipes whose inputs are unavailable (e.g. an all-null base column
        # was dropped); downstream feature-column checks will report the gap.
        if (r["a"] not in new_cols and r["a"] not in df.columns) or (
            r["b"] not in new_cols and r["b"] not in df.columns
        ):
            continue
        a = new_cols[r["a"]] if r["a"] in new_cols else df[r["a"]].to_numpy(dtype=float)
        b = new_cols[r["b"]] if r["b"] in new_cols else df[r["b"]].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            vals = OPS[r["op"]](a, b)
        new_cols[r["name"]] = np.where(np.isfinite(vals), vals, np.nan)
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def _masked_corr(cands: np.ndarray, y: np.ndarray) -> np.ndarray:
    """|corr| of each candidate column with y, using per-column finite rows."""
    mask = np.isfinite(cands)
    x = np.where(mask, cands, 0.0)
    yy = np.where(mask, y[:, None], 0.0)
    n = mask.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        sx, sy = x.sum(axis=0), yy.sum(axis=0)
        sxx, syy, sxy = (x * x).sum(axis=0), (yy * yy).sum(axis=0), (x * yy).sum(axis=0)
        cov = sxy - sx * sy / n
        var_x = sxx - sx * sx / n
        var_y = syy - sy * sy / n
        corr = np.abs(cov / np.sqrt(var_x * var_y))
    corr[(n < _MIN_VALID_ROWS) | ~np.isfinite(corr)] = 0.0
    return corr


def _corr_pair(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < _MIN_VALID_ROWS:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        c = np.corrcoef(a[mask], b[mask])[0, 1]
    return abs(c) if np.isfinite(c) else 0.0


def generate_features(
    train_df: pd.DataFrame,
    y_train: np.ndarray,
    base_cols: list[str],
    n_iterations: int = 2,
    min_target_corr: float = 0.02,
    max_inter_corr: float = 0.95,
    max_new_per_iter: int = 200,
    max_screened: int = 3000,
) -> list[dict]:
    """Return ordered recipes for the kept generated features.

    Only train_df / y_train are used, so the generator never sees test data.
    max_screened bounds the greedy redundancy pass: only the top-N candidates
    by |target corr| are examined each iteration.
    """
    y = np.asarray(y_train, dtype=float)
    pool = {c: train_df[c].to_numpy(dtype=float) for c in base_cols}
    pool_names = list(base_cols)
    n_pairs_done = 0  # pairs among the first `n_pairs_done` names are already screened

    all_recipes = []
    for it in range(1, n_iterations + 1):
        names = pool_names
        # Pairs not yet screened: at least one member beyond the previous pool.
        pairs = [
            (i, j)
            for i in range(len(names))
            for j in range(i + 1, len(names))
            if j >= n_pairs_done
        ]
        n_pairs_done = len(names)
        n_cand = len(pairs) * len(OPS)
        print(f"[featgen] iter {it}: pool={len(names)} pairs={len(pairs)} candidates={n_cand}")
        if not pairs:
            break

        mat = np.column_stack([pool[n] for n in names])
        # Pass 1: |corr(candidate, target)| for every (pair, op), in chunks.
        scored = []  # (abs_corr, i, j, op)
        for op_name, op in OPS.items():
            for start in range(0, len(pairs), _CHUNK):
                chunk = pairs[start:start + _CHUNK]
                ii = [p[0] for p in chunk]
                jj = [p[1] for p in chunk]
                with np.errstate(divide="ignore", invalid="ignore"):
                    cands = op(mat[:, ii], mat[:, jj])
                cands[~np.isfinite(cands)] = np.nan
                corrs = _masked_corr(cands, y)
                keep = np.nonzero(corrs >= min_target_corr)[0]
                scored.extend((corrs[k], *chunk[k], op_name) for k in keep)
        print(f"[featgen] iter {it}: {len(scored)} candidates pass |corr| >= {min_target_corr}")

        # Pass 2: greedy pick by |target corr| with pairwise redundancy check.
        scored.sort(key=lambda t: -t[0])
        kept_vals, kept_recipes = [], []
        for corr, i, j, op_name in scored[:max_screened]:
            if len(kept_recipes) >= max_new_per_iter:
                break
            with np.errstate(divide="ignore", invalid="ignore"):
                vals = OPS[op_name](pool[names[i]], pool[names[j]])
            vals = np.where(np.isfinite(vals), vals, np.nan)
            if any(_corr_pair(vals, kv) >= max_inter_corr for kv in kept_vals):
                continue
            name = f"g{it}_{op_name}__{names[i]}__{names[j]}"
            kept_vals.append(vals)
            kept_recipes.append({"name": name, "op": op_name, "a": names[i], "b": names[j],
                                 "train_corr": round(float(corr), 6)})

        print(f"[featgen] iter {it}: kept {len(kept_recipes)} new features")
        if not kept_recipes:
            break
        for r, v in zip(kept_recipes, kept_vals):
            pool[r["name"]] = v
            pool_names.append(r["name"])
        all_recipes.extend(kept_recipes)

    return all_recipes
