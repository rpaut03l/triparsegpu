"""Standard log-parsing accuracy metrics.

GA  (Grouping Accuracy, Zhu et al. 2019): fraction of lines whose predicted
    group exactly equals its ground-truth group.
PA  (Parsing Accuracy): fraction of lines whose predicted template string
    matches the ground-truth template exactly.
FGA (F1 Grouping Accuracy, LogHub-2.0): F1 over groups treated as sets.
FTA (F1 Template Accuracy): F1 over groups, counting a group correct only if
    its template string also matches.
parity: adjusted Rand index against the Drain3 reference clustering.
"""
import pandas as pd
from sklearn.metrics import adjusted_rand_score


def _groups(ids):
    s = pd.Series(list(ids))
    return {frozenset(idx) for idx in s.groupby(s).groups.values()}


def grouping_accuracy(gt_ids, pred_ids):
    gt, pr = pd.Series(list(gt_ids)), pd.Series(list(pred_ids))
    assert len(gt) == len(pr)
    correct = 0
    pr_sizes = pr.value_counts()
    for _, idx in gt.groupby(gt).groups.items():
        vals = pr[idx]
        first = vals.iloc[0]
        # every line in this GT group has the same predicted id, AND that
        # predicted id contains no lines from other GT groups
        if vals.nunique() == 1 and pr_sizes[first] == len(idx):
            correct += len(idx)
    return correct / len(gt)


def parsing_accuracy(gt_templates, pred_templates):
    gt, pr = list(gt_templates), list(pred_templates)
    assert len(gt) == len(pr)
    return sum(a == b for a, b in zip(gt, pr)) / len(gt)


def fga(gt_ids, pred_ids):
    g, p = _groups(gt_ids), _groups(pred_ids)
    tp = len(g & p)
    if not p or not g or tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec)


def fta(gt_ids, pred_ids, gt_templates, pred_templates):
    """Like FGA but a matched group counts only if its template string agrees."""
    gt_s, pr_s = pd.Series(list(gt_ids)), pd.Series(list(pred_ids))
    gtt, prt = list(gt_templates), list(pred_templates)
    g = {frozenset(i): gtt[list(i)[0]] for i in _groups(gt_ids)}
    p = {frozenset(i): prt[list(i)[0]] for i in _groups(pred_ids)}
    tp = sum(1 for k, v in p.items() if k in g and g[k] == v)
    if not p or not g or tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec)


def drain3_parity(ref_ids, pred_ids):
    return adjusted_rand_score(list(ref_ids), list(pred_ids))


def evaluate(gt_ids, pred_ids, gt_templates=None, pred_templates=None, ref_ids=None):
    out = {"GA": grouping_accuracy(gt_ids, pred_ids), "FGA": fga(gt_ids, pred_ids)}
    if gt_templates is not None and pred_templates is not None:
        out["PA"] = parsing_accuracy(gt_templates, pred_templates)
        out["FTA"] = fta(gt_ids, pred_ids, gt_templates, pred_templates)
    if ref_ids is not None:
        out["parity_ARI"] = drain3_parity(ref_ids, pred_ids)
    return out
