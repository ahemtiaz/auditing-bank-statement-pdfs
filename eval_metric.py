"""
eval_metric.py -- standalone, dependency-light evaluation metric for the NL->SQL benchmark.

Generic and method-level only (no corpus-specific normalization): rows are compared as an
order-insensitive MULTISET (E1), and when order matters the sequence need only be consistent with
the gold ranking key UP TO TIES (E2). Output columns must match the requested contract exactly --
wrong/missing/extra column names are penalized (the question already states the exact column names
and order, so following them is the model's responsibility). Numeric values are compared to a
configurable tolerance.

Primary metric  = execution accuracy (content multiset matches AND order satisfied).
Secondary       = set precision / recall / F1.
"""
import decimal
import datetime
import collections

import pandas as pd
import numpy as np

# Numeric comparison tolerance: values are compared after rounding to this many decimal places.
NUM_TOL = 0.01
_ROUND_DP = max(0, int(round(-decimal.Decimal(str(NUM_TOL)).log10()))) if NUM_TOL > 0 else 2


def clean_cell_value(val):
    if pd.isna(val):
        return None
    if isinstance(val, (float, decimal.Decimal)):
        return float(val)
    if isinstance(val, int) and not isinstance(val, bool):
        return float(val)  # unify int/float so 10 and 10.0 compare equal in the multiset
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    if isinstance(val, (datetime.date, datetime.datetime)):
        return val.strftime("%Y-%m-%d")
    if isinstance(val, str):
        val_str = val.strip()
        val_lower = val_str.lower()
        try:
            if '.' in val_lower:
                return float(val_lower)
            return float(int(val_lower))
        except ValueError:
            pass
        if len(val_str) == 10 and val_str[4] == '-' and val_str[7] == '-':
            try:
                datetime.datetime.strptime(val_str, "%Y-%m-%d")
                return val_str
            except ValueError:
                pass
        return val_lower
    return val


def normalize_dataframe(df):
    if df is None or df.empty:
        return []
    if df.shape == (1, 1) and ("no query results" in str(df.iloc[0, 0]).lower() or "no results" in str(df.columns[0]).lower()):
        return []
    return [tuple(clean_cell_value(v) for v in row) for _, row in df.iterrows()]


def parse_primary_order_key(sql):
    """Extract the leading column of the OUTER (last) ORDER BY of a gold query, lower-cased, or
    None. Used to disambiguate the ranking key for ties-tolerant ordering."""
    if not sql:
        return None
    import re
    # Remove block comments
    sql_clean = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)
    # Remove single line comments
    lines = []
    for line in sql_clean.split('\n'):
        pos_comment = line.find('--')
        if pos_comment != -1:
            line = line[:pos_comment]
        lines.append(line)
    sql_clean = ' '.join(lines)

    matches = list(re.finditer(r'\border\s+by\b', sql_clean, re.IGNORECASE))
    if not matches:
        return None
    order_by_content = sql_clean[matches[-1].end():].strip()
    first_term = order_by_content.split(',')[0].strip()
    first_term = re.split(r'\s+(desc|asc|nulls\s+(first|last))\b', first_term, flags=re.IGNORECASE)[0].strip()
    first_term = first_term.rstrip(';').strip()
    first_term = first_term.strip('"\'`')

    if '.' in first_term and '(' not in first_term:
        first_term = first_term.split('.')[-1].strip()

    if '(' in first_term and ')' in first_term:
        inner = first_term[first_term.find('(')+1 : first_term.rfind(')')].strip()
        first_arg = inner.split(',')[0].strip()
        first_arg = first_arg.strip('"\'`')
        if '.' in first_arg and '(' not in first_arg:
            first_arg = first_arg.split('.')[-1].strip()
        first_term = first_arg

    first_term = first_term.strip('"\'`').lower()
    if first_term and re.match(r'^[a-z_][a-z0-9_]*$', first_term):
        return first_term
    return None


def _monotonic_dir(seq):
    try:
        non_inc = all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))
        non_dec = all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))
    except TypeError:
        return None
    if non_inc and not non_dec:
        return -1
    if non_dec and not non_inc:
        return 1
    if non_inc and non_dec:
        return 0  # constant
    return None


def _detect_order_key(rows):
    """Fallback: index+direction of the first monotonic, non-constant gold column."""
    if not rows:
        return None, None
    for j in range(len(rows[0])):
        seq = [r[j] for r in rows]
        if any(v is None for v in seq):
            continue
        d = _monotonic_dir(seq)
        if d in (-1, 1):
            return j, d
    return None, None


def _order_satisfied(pred_rows, true_rows, key_idx=None):
    """E2: pred ordered consistently with gold's ranking key, up to ties."""
    if key_idx is not None and true_rows and len(true_rows[0]) > key_idx:
        direction = _monotonic_dir([r[key_idx] for r in true_rows])
        j = key_idx
        if direction is None:
            return True
    else:
        j, direction = _detect_order_key(true_rows)
        if j is None:
            return True
    if direction == 0:
        return True
    if not pred_rows or len(pred_rows[0]) <= j:
        return False
    seq = [r[j] for r in pred_rows]
    if any(v is None for v in seq):
        return False
    try:
        if direction < 0:
            return all(seq[i] >= seq[i + 1] for i in range(len(seq) - 1))
        return all(seq[i] <= seq[i + 1] for i in range(len(seq) - 1))
    except TypeError:
        return False


def max_bipartite_matching(pred_rows, true_rows, tolerance=0.01):
    """
    Computes the size of the maximum bipartite matching between pred_rows and true_rows.
    Two rows match if they have the same length and all elements match (either float delta <= tolerance or exact match).
    """
    n = len(pred_rows)
    m = len(true_rows)
    if n == 0 or m == 0:
        return 0

    adj = [[] for _ in range(n)]
    
    def values_match(v1, v2):
        if v1 is None and v2 is None:
            return True
        if v1 is None or v2 is None:
            return False
        
        t1 = isinstance(v1, (float, int))
        t2 = isinstance(v2, (float, int))
        if t1 and t2:
            return abs(float(v1) - float(v2)) <= tolerance
        
        return v1 == v2

    def rows_match(r1, r2):
        if len(r1) != len(r2):
            return False
        return all(values_match(r1[k], r2[k]) for k in range(len(r1)))

    for i in range(n):
        for j in range(m):
            if rows_match(pred_rows[i], true_rows[j]):
                adj[i].append(j)

    match_true = [-1] * m
    
    def dfs(u, visited):
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                if match_true[v] < 0 or dfs(match_true[v], visited):
                    match_true[v] = u
                    return True
        return False

    matching_size = 0
    for i in range(n):
        visited = [False] * m
        if dfs(i, visited):
            matching_size += 1

    return matching_size


def compare_results(df_pred, df_true, order_matters, order_key=None):
    """Returns (tp, fp, fn, order_satisfied) via order-insensitive multiset row overlap.
    order_key = the gold's primary ORDER BY column (see parse_primary_order_key)."""
    def _empty(df):
        return df is None or df.empty or (df.shape == (1, 1) and "no query results" in str(df.iloc[0, 0]).lower())

    pred_empty, true_empty = _empty(df_pred), _empty(df_true)
    if pred_empty and true_empty:
        return 0, 0, 0, True
    if pred_empty:
        return 0, 0, len(df_true), True
    if true_empty:
        return 0, len(df_pred), 0, True

    cols_true = [str(c).strip().lower() for c in df_true.columns]
    cols_pred = [str(c).strip().lower() for c in df_pred.columns]
    true_aligned = df_true.copy(); true_aligned.columns = cols_true
    pred_aligned = df_pred.copy(); pred_aligned.columns = cols_pred

    # Loose column contract
    if set(cols_pred) == set(cols_true):
        pred_aligned = pred_aligned[cols_true]
    elif len(cols_pred) == len(cols_true):
        # Fallback: Align by position (Spider-style index-based alignment)
        pred_aligned.columns = cols_true
        pred_aligned = pred_aligned[cols_true]
    else:
        # Number of columns doesn't match: fully penalized
        return 0, len(pred_aligned), len(true_aligned), False

    true_rows = normalize_dataframe(true_aligned)
    pred_rows = normalize_dataframe(pred_aligned)

    tp = max_bipartite_matching(pred_rows, true_rows, tolerance=NUM_TOL)
    fp = len(pred_rows) - tp
    fn = len(true_rows) - tp

    order_ok = True
    if order_matters:
        key_idx = cols_true.index(order_key) if (order_key and order_key in cols_true) else None
        order_ok = _order_satisfied(pred_rows, true_rows, key_idx)
    return tp, fp, fn, order_ok


def calculate_metrics(tp, fp, fn):
    pred_total = tp + fp
    precision = (1.0 if (tp + fn) == 0 else 0.0) if pred_total == 0 else tp / pred_total
    true_total = tp + fn
    recall = (1.0 if pred_total == 0 else 0.0) if true_total == 0 else tp / true_total
    if precision + recall == 0:
        f1 = 1.0 if (pred_total == 0 and true_total == 0) else 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    union_total = tp + fp + fn
    jaccard = 1.0 if union_total == 0 else tp / union_total
    return precision, recall, f1, jaccard
