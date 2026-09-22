"""Exact shared-window deletion certificates for a frozen neural/evidence score.

No held-out labels enter this module. All arithmetic is float64. The tie rule is the
smallest entity ID. ``radius`` is the largest guaranteed deletion budget and
is capped at the number of available windows; ``min_delete=B+1`` means no
deletion of the available windows changes the answer. Both fixed-denominator
and renormalized evidence channels are implemented. Complete deletion in the
renormalized channel falls back to the frozen neural anchor.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import numpy as np
from scipy.special import softmax


@dataclass
class WindowEvidence:
    objects: np.ndarray
    windows: np.ndarray
    mass: np.ndarray  # windows x supported objects
    reference_mass: float
    event_count: int
    counts: np.ndarray | None = None

    def totals(self, n_entities: int) -> np.ndarray:
        out = np.zeros(n_entities, dtype=np.float64)
        if self.reference_mass > 0:
            out[self.objects] = self.mass.sum(axis=0) / self.reference_mass
        return out


def make_index(events):
    index = {}
    for s, r, o, t in events:
        index.setdefault((int(s), int(r)), []).append((int(t), int(o)))
    return {k: np.asarray(sorted(v), dtype=np.int64) for k, v in index.items()}


def evidence_at(index, subject, relation, timestamp, width=7, offset=0, half_life=35., excluded=()):
    """Only published training events at or before the query time are evidence."""
    source = index.get((subject, relation), np.empty((0, 2), dtype=np.int64))
    source = source[:np.searchsorted(source[:, 0], timestamp, side='right')]
    if len(excluded):source=source[~np.isin(source[:,1],excluded)]
    if not len(source):
        return WindowEvidence(np.array([], dtype=np.int64), np.array([], dtype=np.int64),
                              np.zeros((0, 0)), 0., 0)
    objects, oi = np.unique(source[:, 1], return_inverse=True)
    windows, wi = np.unique((source[:, 0] - offset) // width, return_inverse=True)
    weights = np.exp2(-(timestamp - source[:, 0]) / half_life)
    mass = np.zeros((len(windows), len(objects)), dtype=np.float64)
    np.add.at(mass, (wi, oi), weights)
    counts=np.zeros_like(mass,dtype=np.int64);np.add.at(counts,(wi,oi),1)
    return WindowEvidence(objects, windows, mass, float(weights.sum()), len(source),counts)


def neural_probability(logits, temperature=1.):
    return softmax(np.asarray(logits, dtype=np.float64) / temperature)


def fused_score(probability, evidence, weight=.25, normalization='fixed'):
    if normalization=='renormalized' and evidence.reference_mass==0:
        return probability.copy()
    return (1. - weight) * probability + weight * evidence.totals(len(probability))


def edited_score(score, evidence, weight, deleted_rows, normalization='fixed', probability=None):
    if normalization=='renormalized':
        if probability is None:raise ValueError('Renormalization requires the frozen neural probability.')
        remaining=evidence.mass.copy()
        remaining[list(deleted_rows)]=0.
        total=float(remaining.sum())
        if total>0:
            out=(1-weight)*probability.copy()
            out[evidence.objects]+=weight*remaining.sum(axis=0)/total
        else:out=probability.copy()
        out[~np.isfinite(score)]=-np.inf
        return out
    out = np.array(score, dtype=np.float64, copy=True)
    if evidence.reference_mass > 0 and len(deleted_rows):
        out[evidence.objects] -= weight * evidence.mass[list(deleted_rows)].sum(axis=0) / evidence.reference_mass
    return out


def relevant_competitors(score, evidence, winner, sparse=True):
    """Unsupported competitors share zero editable mass: only the best matters."""
    if not sparse:
        return np.delete(np.arange(len(score)), winner)
    ids = set(map(int, evidence.objects))
    mask = np.ones(len(score), dtype=bool)
    mask[evidence.objects] = False
    mask[winner] = False
    outside = np.flatnonzero(mask)
    if len(outside):
        ids.add(int(outside[np.argmax(score[outside])]))
    ids.discard(winner)
    return np.array(sorted(ids), dtype=np.int64)


def certify(score, evidence, weight=.25, budgets=(1, 2, 3), sparse=True,
            normalization='fixed', probability=None):
    """Return exact-arithmetic radius and a worst-case witness for each budget.

    Positive signed loss is used because the adversary may delete *at most* k
    windows. Ties are checked using entity IDs, not by treating every tie as a
    flip. The sparse path is mathematically equivalent to full enumeration.
    """
    score = np.asarray(score, dtype=np.float64)
    if normalization not in ('fixed','renormalized'):raise ValueError(normalization)
    if normalization=='renormalized' and probability is None:raise ValueError('Missing frozen neural anchor.')
    winner = int(np.argmax(score))
    count = len(evidence.windows)
    ids = relevant_competitors(score, evidence, winner, sparse)
    normalized = (weight / evidence.reference_mass * evidence.mass
                  if evidence.reference_mass > 0 else evidence.mass)
    window_fraction=evidence.mass.sum(axis=1)/(evidence.reference_mass or 1.)
    positions = {int(o): i for i, o in enumerate(evidence.objects)}
    zero = np.zeros(count)
    winner_mass = normalized[:, positions[winner]] if winner in positions else zero
    min_delete = count + 1
    out = {'winner': winner, 'windows': count, 'competitors': len(ids)}
    residual = {k: float('inf') for k in budgets}
    witness = {k: [] for k in budgets}
    adversaries = {k: -1 for k in budgets}
    vulnerable = {k: False for k in budgets}
    for competitor in ids:
        if not np.isfinite(score[competitor]):continue
        other = normalized[:, positions[int(competitor)]] if int(competitor) in positions else zero
        delta = winner_mass - other
        if normalization=='renormalized' and evidence.reference_mass>0:
            delta=delta+(1-weight)*(probability[winner]-probability[competitor])*window_fraction
        complete_order=np.argsort(-delta,kind='stable')
        order = complete_order[delta[complete_order] > 0]
        # The ratio is defined for proper subsets. The all-deleted fallback is
        # evaluated separately, avoiding multiplication by a zero denominator.
        if normalization=='renormalized':order=order[:max(0,count-1)]
        loss = np.r_[0., np.cumsum(delta[order])]
        gap = score[winner] - score[competitor]
        flips = loss >= gap if competitor < winner else loss > gap
        surplus=gap-loss
        if normalization=='renormalized' and count:
            # Suffix sums preserve old, tiny weights when a deletion removes
            # almost all recent mass. Subtracting removed mass from the total
            # would lose that information through catastrophic cancellation.
            def suffix(x):return np.r_[np.cumsum(x[complete_order][::-1])[::-1],0.][:len(order)+1]
            keep_mass=suffix(window_fraction)
            keep_w,keep_c=suffix(winner_mass),suffix(other)
            direct_w=(1-weight)*probability[winner]+keep_w/keep_mass
            direct_c=(1-weight)*probability[competitor]+keep_c/keep_mass
            flips=direct_c>=direct_w if competitor<winner else direct_c>direct_w
            surplus=((1-weight)*(probability[winner]-probability[competitor])*keep_mass+keep_w-keep_c)
        # Near cancellation, use the same direct score subtraction as inference.
        # This avoids declaring a numerical flip from differently rounded sums.
        for j in (np.flatnonzero(np.abs(loss-gap)<=2e-12) if normalization=='fixed' else []):
            direct = edited_score(score, evidence, weight, order[:j],normalization,probability)
            flips[j] = (direct[competitor] > direct[winner] or
                        (direct[competitor] == direct[winner] and competitor < winner))
        where = np.flatnonzero(flips)
        if len(where):
            min_delete = min(min_delete, int(where[0]))
        for k in budgets:
            take = min(k, len(order))
            value = float(surplus[take])
            flip = bool(flips[take])
            if value < residual[k] or (value == residual[k] and flip and not vulnerable[k]):
                residual[k] = value
                witness[k] = order[:take].tolist()
                adversaries[k] = int(competitor)
                vulnerable[k] = flip
    if normalization=='renormalized':
        fallback=probability.copy();fallback[~np.isfinite(score)]=-np.inf
        new_winner=int(np.argmax(fallback))
        if new_winner!=winner:
            min_delete=min(min_delete,count)
            for k in budgets:
                if k>=count:
                    vulnerable[k]=True
                    witness[k]=list(range(count))
                    adversaries[k]=new_winner
                    residual[k]=min(residual[k],float((1-weight)*(fallback[winner]-fallback[new_winner])))
    out.update(min_delete=min_delete, radius=min(min_delete - 1, count),
               all_stable=bool(min_delete == count + 1))
    for k in budgets:
        out[f'residual{k}'] = residual[k]
        out[f'robust{k}'] = not vulnerable[k]
        out[f'witness{k}'] = witness[k]
        out[f'adversary{k}'] = adversaries[k]
    return out


def brute_force(score, evidence, weight, budget, normalization='fixed', probability=None):
    """Independent exponential oracle for small, meaningful property tests."""
    winner = int(np.argmax(score))
    for k in range(min(budget, len(evidence.windows)) + 1):
        for rows in combinations(range(len(evidence.windows)), k):
            if int(np.argmax(edited_score(score, evidence, weight, rows,normalization,probability))) != winner:
                return False
    return True
