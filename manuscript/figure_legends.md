# Current figure and table captions

\textbf{The StableKG decision layer.} (a) Evidence for a candidate object is time-local and receives a decaying weight. (b) Calibrated belief and the deletion diagnostic describe different aspects of a prediction; all 5,000 ICEWS14 test queries are shown, and the shaded region marks the seed-0 acceptance threshold selected at a 20\% validation target. (c) A changed fact is propagated through the reverse dependency closure, so only affected derived summaries and answers are updated.

Causal filtered ranking on the public test queries. Ranking is deterministic once the evidence index is fixed.

\textbf{Selective risk--coverage curves.} Each panel reports the test risk after sorting predictions by the belief-only baseline or the StableKG belief. Curves, points and AURC use calibration seed 0. Points show the actual test coverage and risk obtained at thresholds selected for 20\% and 40\% validation coverage; Tables S1 and S2 report seed aggregates and uncertainty.

\textbf{Reliability of predicted belief.} Empirical accuracy is plotted against predicted belief in ten bins using all 5,000 test queries and calibration seed 0. The diagonal is perfect calibration. StableKG follows the diagonal more closely on the two ICEWS datasets; the large-scale GDELT panel shows a near-neutral calibration change.

Neural-backbone confirmation. Values are means over three training seeds. StableKG is applied after the backbone score and uses the same validation-only operating protocol.

Recent author-code backbone comparison. B is the validation-selected calibrated-score comparator and S is StableKG.

\textbf{Temporal stress tests.} (a,b) Risk under a frozen 20\% validation-target threshold across chronological timestamp blocks for ICEWS14 and ICEWS05-15. (c,d) Realized test coverage in the same blocks for each selector. Horizontal dotted lines mark the 20\% validation target. All four panels use chronological predict-before-reveal outputs; thresholds remain fixed throughout test replay.

\textbf{Evidence interventions.} Additive-scorer winner-flip rates after a single-event deletion, deletion of the strongest seven-unit winner window, or injection of one current-time counter-event. Bars show the lower and upper certificate quartiles; error bars are bootstrap 95\% intervals over queries. Panels a--c show flip rates for the low/high deletion-diagnostic groups. Panels d--f compare AUROC for predicting survival across the same edits; cells use all supported queries, and darker shading denotes higher AUROC. Deletion score denotes $c$, normalized margin denotes $z$, and raw margin denotes $\Delta$. Query counts are in Supplementary Table~S4.

Matched feature controls. AURC averaged over five validation permutations; lower is better. B contains margin, log-support and recency. $c$ is the deletion diagnostic score, $z$ the normalized margin, and $(f,d)$ the removal cost and diversity.

End-to-end public-graph update maintenance. Means over 10 repetitions; full refresh re-evaluates every registered decision from the same indexed summaries.

\textbf{Exact incremental updates.} (a) Fractional reduction in dependency nodes visited by the closure update. (b) Fractional reduction in propagation time with a precomputed closure. Bars show means over 20 controlled seeds and error bars show 95\% bootstrap intervals. Full and incremental recomputation agree exactly in all seeds.