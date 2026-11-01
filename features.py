"""BN-1: offline Bayesian-network feature selection, learned on training data only.

Structure: PC-stable algorithm (chi-square CI tests, alpha=0.01) on day-of-year
deseasonalised anomalies discretised into training-period tertiles. Using anomalies
prevents spurious edges created by the shared annual cycle. The selected inputs are
the Markov blanket of the target (Kt anomaly). Parameters: BDeu, equivalent sample
size N'=5. Every quantity (deseasonalisation, bin edges, CI tests, CPTs) is computed
from data strictly before the forecast origin.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logging.getLogger("pgmpy").setLevel(logging.ERROR)


def discretise(train: pd.DataFrame, n_bins: int = 3) -> tuple[pd.DataFrame, dict]:
    out, edges = pd.DataFrame(index=train.index), {}
    labels = ["low", "medium", "high"]
    for c in train.columns:
        q = np.unique(np.nanquantile(train[c], np.linspace(0, 1, n_bins + 1)))
        q[0], q[-1] = -np.inf, np.inf
        edges[c] = q
        out[c] = pd.cut(train[c], q, labels=labels[: len(q) - 1] if len(q) - 1 == 3 else None,
                        include_lowest=True).astype(str)
    return out, edges


def learn_bn1(anom_train: pd.DataFrame, target: str = "Kt", alpha: float = 0.01, ess: float = 5.0):
    from pgmpy.estimators import PC, BayesianEstimator
    try:
        from pgmpy.models import DiscreteBayesianNetwork as BN
    except ImportError:  # pgmpy < 1.0
        from pgmpy.models import BayesianNetwork as BN

    disc, edges = discretise(anom_train.dropna())
    skel, sep_sets = PC(disc).estimate(variant="stable", ci_test="chi_square",
                                       significance_level=alpha, return_type="skeleton",
                                       show_progress=False)
    # Selected inputs = adjacency set ("parents-and-children" set) of the target in the
    # PC skeleton: variables that remain dependent on Kt given every conditioning subset.
    pc_set = sorted(skel.neighbors(target)) if target in skel.nodes else []
    # Discriminative structure used only to report P(Kt | selected inputs) (BDeu means)
    model = BN([(v, target) for v in pc_set])
    model.add_nodes_from(disc.columns)
    cols = pc_set + [target]
    cpds = BayesianEstimator(model, disc).get_parameters(prior_type="BDeu", equivalent_sample_size=ess)
    model.add_cpds(*(cpds.values() if isinstance(cpds, dict) else cpds))
    return dict(skeleton=skel, markov_blanket=pc_set, model=model, edges=edges, data=disc,
                sep_sets=sep_sets)


def cpt_table(bn: dict, target: str = "Kt") -> pd.DataFrame:
    """P(target | parents) as a long table (BDeu posterior means) for reporting."""
    cpd = bn["model"].get_cpds(target)
    vals = cpd.get_values()
    states = cpd.state_names
    parents = cpd.variables[1:]
    import itertools
    rows = []
    combos = list(itertools.product(*[states[p] for p in parents])) if parents else [()]
    for j, combo in enumerate(combos):
        row = {p: s for p, s in zip(parents, combo)}
        for i, s in enumerate(states[target]):
            row[f"P({target}={s})"] = vals[i, j]
        rows.append(row)
    return pd.DataFrame(rows)
