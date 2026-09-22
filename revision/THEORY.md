# Renormalized shared-window certificate

This extension was derived and property-tested before any revised test scores
were generated. It supersedes the fixed-reference channel as the primary
method; the fixed-reference variant remains a controlled ablation.

Let the frozen admissible neural distribution be p, the evidence weight be
0 <= lambda < 1, and e_bo >= 0 denote object o's mass in nonempty window b.
S=sum_bo e_bo, m_b=sum_o e_bo. For a proper deleted subset D, define

g_D(o) = (1-lambda) p_o + lambda (E_o - sum_{b in D} e_bo)
                                      / (S - sum_{b in D} m_b).

For no remaining evidence, set g_D=p. This is a genuine mixture distribution
over the fixed admissible candidate vocabulary. Training-observed answers are
excluded from that vocabulary and from the evidence channel before evaluation;
held-out truth labels do not construct it.

For the original winner w and competitor c, put

G_c = g_empty(w)-g_empty(c),
d_bc = lambda(e_bw-e_bc)/S + (1-lambda)(p_w-p_c)m_b/S.

Multiplying the post-edit gap by the positive retained-mass fraction gives

(1 - sum_{b in D}m_b/S) [g_D(w)-g_D(c)] = G_c-sum_{b in D}d_bc.

Therefore an at-most-k deletion adversary on proper subsets chooses the largest
min(k,B-1) positive d_bc values. This yields the exact sign of the worst possible
pairwise margin, without enumerating subsets. If k>=B, also check the complete-
deletion fallback p. The answer is certified iff no competitor can overtake it,
including the fixed smallest-entity-ID tie rule. A negative surplus provides
a constructive witness; a zero surplus is resolved by the tie rule. The numeric
implementation directly re-evaluates near-cancellation witnesses in float64.

The surplus is a *sign certificate*, not the unscaled numerical worst-case
post-edit margin. Do not describe it as the latter: retained mass varies by D.

## Sparse exact reduction
For any two candidates with zero evidence, their post-edit score difference is
(1-lambda)(p_c-p_c') for every D, including the empty-evidence fallback (where
the factor is one). Thus only the best neural candidate outside evidence support
can be the most dangerous unsupported competitor. Check supported competitors
plus this single representative. This is exact for the full entity vocabulary.

After the neural score vector is available, identify its maximum and the best
unsupported candidate in O(N); certify C supported competitors in
O(C B log B), with O(CB) window statistics. Brute force would enumerate a number
of subsets exponential in B. A full competitor implementation uses O(N B log B).

## Update locality
With encoder parameters, query times and admissible vocabulary frozen, an event
(s,r,o,t) affects only registered queries with the same (s,r), query time >= t,
and admissible object o. Update one cell and the total mass in each affected
query cache; refresh the winner and certificate. Every other score/certificate
is unchanged. Re-evaluate affected queries from their updated sufficient
statistics; the certificate is computed for their new winner. There is no
claim to update neural parameters or to unlearn a deleted training fact.

## Controls
Fixed denominator: omit the second term of d_bc; no renormalization after edit.
Conservative winner-only bound for the renormalized channel: replace
lambda(e_bw-e_bc)/S by lambda e_bw/S, retaining the neural normalization term.
It upper-bounds the true signed loss, so its positive-part top-k sum is a
sufficient but generally non-sharp condition. Check the same full-deletion
fallback in both certificates. Random deletion probes are empirical diagnostics,
never labelled mathematical certificates.
