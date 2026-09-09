"""Populate the revised narrative and SI directly from the corrected release."""
from pathlib import Path
import json
import re
import shutil
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "manuscript"
BACKUP = PAPER / "archive_before_review_20260909"
BACKUP.mkdir(exist_ok=True)
for name in ("main.tex", "main.pdf", "supplementary.tex", "supplementary.pdf", "references.bib"):
    if not (BACKUP / name).exists():
        shutil.copy2(PAPER / name, BACKUP / name)
names = ("ICEWS14", "ICEWS05-15", "GDELT")
pub = {n: json.loads((ROOT / f"results_final_v5/{n}_summary.json").read_text()) for n in names}
neural = {n: pd.read_csv(ROOT / f"results_neural_v5/{n}_seeds.csv").mean(numeric_only=True) for n in names[:2]}
chrono = {n: json.loads((ROOT / f"results_chronological_v5/{n}_chronological_summary.json").read_text()) for n in names[:2]}
inter = {n: json.loads((ROOT / f"results_interventions_v5/{n}_interventions.json").read_text()) for n in names}
controls = {n: pd.read_csv(ROOT / f"results_review_20260909/corrected/{n}_seeds.csv").groupby("model").mean(numeric_only=True) for n in names}
paired = pd.read_csv(ROOT / "results_final_v5/analysis/paired_block_bootstrap.csv")
fixed = pd.read_csv(ROOT / "results_final_v5/analysis/fixed_count_comparison.csv")
syn = json.loads((ROOT / "results_final_v5/analysis/synthetic_summary_20.json").read_text())
recent = {
    "ICEWS14": pd.read_csv(ROOT / "results_recent_v2/TeRDy/ICEWS14/seed0/metrics.csv"),
    "ICEWS05-15": pd.read_csv(ROOT / "results_recent_v3/TeRDy/ICEWS05-15/seed0/metrics.csv"),
}
updates = {n: pd.read_csv(ROOT / f"results_real_updates/{n}_trials.csv") for n in names}
t = (BACKUP / "main.tex").read_text(encoding="utf-8")
t = t.replace(r"\usepackage{url}", r"\usepackage{xurl}" + "\n" + r"\usepackage[section]{placeins}" + "\n" + r"\renewcommand{\topfraction}{0.9}" + "\n" + r"\renewcommand{\textfraction}{0.08}")
t = t.replace(r"\begin{figure}[t]", r"\begin{figure}[!htbp]")
t = t.replace(r"\begin{table}[t]", r"\begin{table}[!htbp]")
t = t.replace(r"\begin{document}", r"\makeatletter" + "\n" + r"\setlength{\@fptop}{0pt}" + "\n" + r"\setlength{\@fpsep}{16pt}" + "\n" + r"\setlength{\@fpbot}{0pt plus 1fil}" + "\n" + r"\makeatother" + "\n" + r"\begin{document}")

def replace_between(start, end, replacement):
    global t
    left = t.index(start)
    right = t.index(end, left)
    t = t[:left] + replacement.strip() + "\n\n" + t[right:]

def pair(s, metric):
    return f"{s['baseline_' + metric]:.4f} to {s['stability_' + metric]:.4f}"

abstract = r'''\begin{abstract}
Evolving knowledge graphs require a decision beyond ranking missing facts: which predictions remain usable when their supporting evidence changes? We introduce \StableKG{}, a selective reasoning protocol that couples calibrated belief with explicit evidence-deletion diagnostics. Under an additive temporal evidence model, a window-count certificate gives the smallest number of supporting windows whose deletion erases a positive winner margin. The associated removal cost and temporal diversity describe how support is distributed; a validation-trained selector combines them with the normalized evidence margin to decide when to accept or abstain. Across ICEWS14 and ICEWS05-15, the selector reduces area under the risk--coverage curve from @PAIR1@ and from @PAIR2@, respectively. Improvements persist with a learned temporal backbone and when events are revealed chronologically. Matched feature controls identify the normalized margin as a major contributor to selective accuracy, while direct evidence edits establish the diagnostic value of temporal support structure. A separate controlled dependency-graph experiment reduces recomputation work by 71.6\% and propagation time by 47.7\%, preserving exact agreement with full recomputation. By connecting prediction, evidence sensitivity and selective acceptance, \StableKG{} provides an inspectable decision protocol for reasoning over changing knowledge.
\end{abstract}'''.replace("@PAIR1@", pair(pub[names[0]], "aurc")).replace("@PAIR2@", pair(pub[names[1]], "aurc"))
replace_between(r"\begin{abstract}", r"\begin{keyword}", abstract)

intro = r'''\section{Introduction}
A predicted fact can be accurate today and become unreliable after its supporting evidence changes. Knowledge graphs make this problem concrete: event streams continually add relations, revise histories and change the context in which missing facts are inferred. Temporal knowledge graph completion (TKGC) supplies a ranked answer, whereas a downstream decision also needs to determine whether that answer should be accepted and what evidence would prompt its reconsideration. Connecting these decisions requires an explicit account of how a prediction depends on its history.

Recent advances offer increasingly expressive accounts of temporal structure. Learned granularity and multi-granularity histories represent information at different time scales \citep{geng2025granularity,zhu2025multigran}; extrapolated historical facts, dual-view reasoning and information filtering refine which histories enter a prediction \citep{dao2025hgct,chen2025dualview,fernando2025filtering}. Spatio-temporal state embeddings, cross-space representations and query-adaptive relation learning broaden the representation choices \citep{ji2025stse,vu2025tcrosse,guo2025mamba}. The 2026 literature extends this direction through structure-preserving embeddings, sparse fact aggregation and prior-experience completion \citep{xu2026householder,zhu2026splices,peng2026icpe}, alongside semantic dual graphs and hybrid history networks \citep{zhang2026dstag,ye2026dualhistory}. These developments motivate a complementary question: how should a system use the resulting predictions when supporting evidence can be edited or withdrawn?

Confidence and evidential dependence answer different parts of that question. Uncertainty modeling in knowledge graph embeddings addresses predictive uncertainty \citep{liu2025uncertkg}, while label supervision and selective prediction relate uncertainty to correctness and rejection decisions \citep{li2025vicinal,szabadvary2025reject}. Temporal conformal methods address non-exchangeability and forecast uncertainty \citep{wang2025nonexchangeable,zhao2025stgraph}. An evidence-deletion query instead fixes a particular conclusion and asks which changes to its support would erase its advantage over alternatives. A high score alone does not specify this dependence: concentrated support and support distributed across multiple windows can yield the same winner margin.

Temporal reasoning is also becoming more inspectable. Rule validity, logical paths, knowledge-guided editing and semantic dependencies expose different aspects of the reasoning process \citep{pan2025validity,zhang2026logicalpaths,fan2026editing,yu2026dependency}. Graph-text alignment, tensorized relations and global interaction provide further routes to integrating evidence \citep{wan2026tgcallm,lee2026tirano,wang2026globalinteraction}. Cyclic evolution, frequency-aware interaction and in-context contrastive learning enrich the historical context \citep{bi2026globallocal,wang2026multidim,li2026calendar}; robustness studies examine sensitivity and bias in graph completion \citep{li2026entropyrobust,moon2026sharpness}. We focus on the operational connection between this evidence structure and the decision to expose a prediction.

Here we ask whether temporal evidence descriptors can improve selective decisions while making the current answer's response to deletion explicit. \StableKG{} combines three quantities with distinct roles: a discrete window-deletion certificate, continuous evidence diagnostics, and a calibrated probability of correctness. The certificate has an exact interpretation for an additive evidence scorer; the diagnostics also attach to candidates proposed by a learned scorer. We test their value through public point-query benchmarks, matched feature controls, neural transfer, chronological replay and direct interventions. A separate dependency-graph experiment examines the computational benefit of revisiting only affected conclusions. This sequence connects the prediction decision to the evidence that sustains it.
'''
replace_between(r"\section{Introduction}", r"\section{Results}", intro)
replace_between(r"\subsection{A certificate adds", r"\begin{figure}", r'''\subsection{Evidence structure adds an inspectable axis to belief}
Time decay measures age; evidence deletion measures dependence. Under the additive scorer, sorting the support windows by their mass identifies the minimum number of windows needed to erase a positive winner margin. This window count certifies survival against every smaller winner-window deletion. The removal cost and temporal diversity then supply continuous diagnostics for the selector (Fig.~\ref{fig:concept}). Calibrated belief determines acceptance, and the evidence descriptors explain the local support structure accompanying that decision.''')
t = t.replace("the shaded quadrant marks the high-belief, high-certificate operating region", "all 5,000 ICEWS14 test queries are shown, and the shaded region marks the seed-0 acceptance threshold selected at a 20\\% validation target")
t = t.replace("from raw facts to time-window masses", "from inputs to derived quantities")
replace_between(r"\subsection{Causal public benchmarks", r"\begin{table}", r'''\subsection{Evidence-aware selection lowers accepted risk}
We evaluated published $\mathrm{Pe}$ point queries from three TFLEX-format archives: ICEWS14, ICEWS05-15 and GDELT. Training events form the evidence index, with a cutoff at the query timestamp. Each split contributes 5,000 deterministically spaced validation and test queries. The base scorer's filtered ranking establishes the prediction task (Table~\ref{tab:ranking}); the selective layer changes the order in which predictions are accepted without changing the candidate ranking.''')
t = t.replace("Values are means over five calibration seeds; ranking itself is deterministic once the evidence index is fixed.", "Ranking is deterministic once the evidence index is fixed.")
paragraph = "The evidence-aware selector improved the full risk--coverage ordering on both ICEWS benchmarks. "
for n in names[:2]:
    s = pub[n]
    q = paired[(paired.dataset == n) & (paired.metric == "aurc_delta")].iloc[0]
    paragraph += f"On {n}, AURC decreased from {pair(s, 'aurc')}; the seed-0 paired timestamp-block difference was {q.estimate:.4f} (95\\% CI {q.ci_low:.4f} to {q.ci_high:.4f}). "
paragraph += "At exactly 20\\% test coverage (1,000 accepted queries per method), risk decreased from "
paragraph += " and from ".join(f"{fixed[(fixed.dataset==n)&(fixed.coverage==.2)].iloc[0].baseline_risk:.4f} to {fixed[(fixed.dataset==n)&(fixed.coverage==.2)].iloc[0].stability_risk:.4f}" for n in names[:2])
paragraph += ", respectively; paired timestamp-block intervals for both risk differences excluded zero (Supplementary Table~S8). "
paragraph += "On GDELT, AURC changed from " + pair(pub["GDELT"], "aurc") + "; the 40\\% operating point showed a small risk increase (Supplementary Table~S1). Figure~\\ref{fig:risk} shows the complete curves; Supplementary Table~S2 reports paired intervals under frozen validation thresholds."
replace_between("The stability selector changes", r"\begin{figure}", paragraph)
t = t.replace("Points mark the operating points closest to 20\\% and 40\\% coverage selected on validation data. AURC is printed inside each panel for the StableKG curve.", "Curves, points and AURC use calibration seed 0. Points show the actual test coverage and risk obtained at thresholds selected for 20\\% and 40\\% validation coverage; Tables S1 and S2 report seed aggregates and uncertainty.")
cal = "The same evidence features also improved probability calibration on the ICEWS datasets. "
for n in names[:2]:
    cal += f"On {n}, Brier score changed from {pair(pub[n], 'brier')} and ten-bin expected calibration error (ECE) from {pair(pub[n], 'ece')}. "
cal += "Figure~\\ref{fig:reliability} displays seed-0 reliability across all test queries; aggregate values for all three datasets are in Supplementary Table~S1."
replace_between("Calibration changes in the same direction", r"\begin{figure}", cal)
t = t.replace("Reliability of the accepted belief.", "Reliability of predicted belief.")
t = t.replace("Empirical accuracy is plotted against predicted belief in ten bins.", "Empirical accuracy is plotted against predicted belief in ten bins using all 5,000 test queries and calibration seed 0.")
neu = r"\subsection{The decision layer transfers to a learned temporal scorer}" + "\n"
neu += "A temporal ComplEx model supplied candidates from learned embeddings, with evidence diagnostics computed for that model's selected object. Across three training seeds, the model achieved MRR "
neu += f"{neural[names[0]]['test_mrr']:.4f} on ICEWS14 and {neural[names[1]]['test_mrr']:.4f} on ICEWS05-15. "
neu += "The selective layer reduced AURC from " + pair(neural[names[0]], "aurc") + " and from " + pair(neural[names[1]], "aurc") + ", respectively (Table~\\ref{tab:neural}). This transfer shows that evidence features can improve acceptance decisions for learned predictions as well as for the additive scorer. Calibration variation across the two settings is reported alongside selective risk."
replace_between(r"\subsection{The effect transfers", r"\begin{table}", neu)
start = t.index(r"\label{tab:neural}")
end = t.index(r"\end{table}", start)
table = r"\label{tab:neural}" + "\n\\small\n\\begin{tabular}{lrrrrr}\n\\toprule\nDataset & MRR & AURC B & AURC S & ECE B & ECE S\\\\\n\\midrule\n"
for n in names[:2]:
    s = neural[n]
    table += f"{n} & {s.test_mrr:.4f} & {s.baseline_aurc:.4f} & {s.stability_aurc:.4f} & {s.baseline_ece:.4f} & {s.stability_ece:.4f}\\\\\n"
table += "\\bottomrule\n\\end{tabular}\n"
t = t[:start] + table + t[end:]
recent_text = r'''\subsection{Recent temporal backbones retain the evidence-aware gain}
To test transfer to a current high-capacity temporal encoder, we ran the public author implementation of TeRDy (ACL 2025) under the same 5,000-query Pe panel, candidate vocabulary and validation-only selection protocol. On ICEWS14 (rank 6,000; 12 epochs), TeRDy reached MRR 0.6571; StableKG reduced AURC from 0.3801 for the calibrated-score comparator to 0.2896 and exact-20\% risk from 0.1880 to 0.1530. On ICEWS05-15, the author default rank 8,000 exceeded the available 8-GB GPU memory; the pre-specified rank-2,000 adaptation completed 8 epochs and reached MRR 0.5355, with AURC changing from 0.3801 to 0.3716 and exact-20\% risk from 0.2330 to 0.2210. The scoring equations and regularizers are unchanged; rank and resource outcomes are recorded with each checkpoint.

\begin{table}[!htbp]
\centering
\caption{Recent author-code backbone comparison. B is the validation-selected calibrated-score comparator and S is StableKG.}
\label{tab:recent}
\small
\resizebox{\linewidth}{!}{%
\begin{tabular}{lrrrrrr}
\toprule
Dataset & Rank & Epochs & MRR & AURC B & AURC S & Risk20 B/S\\
\midrule
ICEWS14 & 6000 & 12 & 0.6571 & 0.3801 & 0.2896 & 0.1880/0.1530\\
ICEWS05-15 & 2000 & 8 & 0.5355 & 0.3801 & 0.3716 & 0.2330/0.2210\\
\bottomrule
\end{tabular}}
\end{table}'''
chron = r"\subsection{Chronological replay preserves the decision advantage}" + "\n"
chron += "We then evaluated decisions before revealing the events they predict. The first 70\\% of unique timestamps supplied the initial history, the next 15\\% supplied validation and the final 15\\% supplied test. Every query at a timestamp was scored before any event at that timestamp entered the index. "
for n in names[:2]:
    chron += f"On {n}, AURC changed from {pair(chrono[n], 'aurc')} and risk near the 20\\% validation target from {pair(chrono[n], 'risk@0.20')}. "
chron += "This replay extends the selective advantage to a strict predict-before-reveal setting (Fig.~\\ref{fig:temporal}a,b). The corresponding realized coverage remains visible across timestamp blocks (panels c,d), connecting the observed risk to the fraction of decisions accepted."
replace_between(r"\subsection{Chronological replay preserves", r"\begin{figure}", recent_text + "\n\n" + chron)
intertext = r"\subsection{Evidence edits expose local decision stability}" + "\n"
intertext += "We recorded the winner and evidence diagnostics before editing each supported test query. Three independent edits removed the most recent winner event, removed its strongest seven-unit support window, or injected a current-time event for the runner-up. Low diagnostic scores identified conclusions more likely to change under each edit (Fig.~\\ref{fig:interventions}). "
for n in names[:2]:
    s = inter[n]
    intertext += f"On {n}, window-deletion flip rates were {s['low_certificate_window_flip']:.4f} in the lower score quartile and {s['high_certificate_window_flip']:.4f} in the upper quartile. "
diagnostics = pd.read_csv(ROOT / "results_review_20260909/corrected/diagnostic_controls.csv")
intertext += "For window survival, the deletion diagnostic achieved AUROC "
intertext += ", ".join(f"{diagnostics[(diagnostics.dataset==n)&(diagnostics.edit=='window')&(diagnostics.feature=='certificate')].iloc[0].survival_auroc:.4f}" for n in names)
intertext += " on ICEWS14, ICEWS05-15 and GDELT, compared with "
intertext += ", ".join(f"{diagnostics[(diagnostics.dataset==n)&(diagnostics.edit=='window')&(diagnostics.feature=='margin')].iloc[0].survival_auroc:.4f}" for n in names)
intertext += " for the raw margin. Thus temporal support structure supplies information specific to whole-window deletion. Under single-event injection, raw margin was the strongest diagnostic, as expected for an additive scorer (Supplementary Table~S7)."
replace_between(r"\subsection{Interventions validate", r"\begin{figure}", intertext)
t = t.replace("seven-day", "seven-unit").replace("Seven-day", "Seven-unit")
t = t.replace("Winner-flip rates after", "Additive-scorer winner-flip rates after")
t = t.replace("The survival correlation printed above each panel is Spearman's $\\rho$ between certificate and non-flip outcome.", "Panels a--c show flip rates for the low/high deletion-diagnostic groups. Panels d--f compare AUROC for predicting survival across the same edits; cells use all supported queries, and darker shading denotes higher AUROC. Deletion score denotes $c$, normalized margin denotes $z$, and raw margin denotes $\\Delta$. Query counts are in Supplementary Table~S4.")
t = t.replace("Risk at approximately 20\\% coverage across chronological timestamp blocks", "Risk under a frozen 20\\% validation-target threshold across chronological timestamp blocks")
t = t.replace("(c,d) A controlled drift stream with repeated evidence changes: accepted risk and coverage for a static score, global time decay and the stability-aware policy. Vertical dotted lines mark drift events.", "(c,d) Realized test coverage in the same blocks for each selector. Horizontal dotted lines mark the 20\\% validation target. All four panels use chronological predict-before-reveal outputs; thresholds remain fixed throughout test replay.")
t = t.replace("Calibrated belief and the intervention-linked stability certificate describe different aspects of a prediction", "Calibrated belief and the deletion diagnostic describe different aspects of a prediction")
abl = r'''\subsection{Matched controls distinguish selection from diagnosis}
We held margin, support, recency, ridge penalty and validation splits fixed while adding or removing evidence features. Adding the diagnostic score improved ICEWS selection, and adding the normalized margin captured most of the full selector's gain. Together with the intervention controls, this separates two useful roles: normalized evidence separation supports correctness ordering, while temporal support structure identifies susceptibility to window deletion. Table~\ref{tab:matched} reports the matched comparisons; the complete 11-model grid, including softmax/entropy and window-count controls, appears in Supplementary Table~S3.

\begin{table}[t]
\centering
\caption{Matched feature controls. AURC averaged over five validation permutations; lower is better. B contains margin, log-support and recency. $c$ is the deletion diagnostic score, $z$ the normalized margin, and $(f,d)$ the removal cost and diversity.}
\label{tab:matched}
\small
\begin{tabular}{lrrr}
\toprule
Features & ICEWS14 & ICEWS05-15 & GDELT\\
\midrule
'''
for key, label in [("margin_only", "Margin only"), ("matched_base", "B"), ("base_plus_c", "B + $c$"), ("base_plus_z", "B + $z$"), ("nonredundant", "B + $z,f,d$"), ("full_original", "Full ($\\mathrm{B}+z,f,d,c$)")]:
    abl += label + " & " + " & ".join(f"{controls[n].loc[key, 'aurc']:.4f}" for n in names) + "\\\\\n"
abl += "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
replace_between(r"\subsection{Ablations identify", r"\subsection{Dependency closure", abl)
replace_between(r"\subsection{Dependency closure", r"\begin{figure}", r'''\subsection{Public-graph updates scale with the affected decision set}
We measured the complete event-to-decision maintenance path on all three public archives. The index stores timestamp multiplicities, seven-unit evidence bins, reverse subject--relation lookup and calibrated acceptance outputs for every registered query key. Insertion, deletion and counter-event batches touched 1, 10, 100 or 1,000 events; a clock advance refreshed all decisions. Across 10 repetitions, incremental outputs matched full indexed refresh and an independent raw-event oracle below $10^{-14}$, including winner identity, certificate count and acceptance. For a batch of 100 events, the incremental path refreshed 98.2, 97.7 and 98.7 decisions on average on ICEWS14, ICEWS05-15 and GDELT, compared with 21,667, 51,292 and 9,741 decisions for full refresh. Mean times were 136.3 to 1.3~ms, 363.6 to 2.2~ms and 223.0 to 4.5~ms, respectively (Table~\ref{tab:publicupdates}). A global clock advance correctly refreshes every decision because decay changes all margins; its measured costs were 126, 359 and 228~ms. Resident index growth was 38.1, 144.3 and 515.9~MB, with the largest value corresponding to the 2.3-million-event GDELT archive.

\begin{table}[!htbp]
\centering
\caption{End-to-end public-graph update maintenance. Means over 10 repetitions; full refresh re-evaluates every registered decision from the same indexed summaries.}
\label{tab:publicupdates}
\small
\resizebox{\linewidth}{!}{%
\begin{tabular}{lrrrrrr}
\toprule
Dataset & Events & Decisions & Batch & Full ms & Incremental ms & Refreshed\\
\midrule
ICEWS14 & 72,826 & 21,667 & 100 & 136.3 & 1.3 & 98.2\\
ICEWS05-15 & 368,962 & 51,292 & 100 & 363.6 & 2.2 & 97.7\\
GDELT & 2,308,165 & 9,741 & 100 & 223.0 & 4.5 & 98.7\\
\bottomrule
\end{tabular}}
\end{table}

\subsection{Dependency closure reduces exact recomputation work}
We evaluated incremental maintenance in a separate controlled rule graph with 220 nodes and 20 independent seeds. The affected closure contained 62.6 nodes on average, reducing visited nodes by 71.6\% (95\% CI 70.7--72.3\%). With the closure cached, propagation time decreased from 0.881 to 0.455 ms, a 47.7\% reduction (95\% CI 42.2--52.8\%). Full and incremental outputs agreed exactly for every seed (Fig.~\ref{fig:efficiency}). This experiment verifies the dependency-closure algorithm for deterministic rule propagation; it measures that computation separately from the public TKGC scorer.''')
t = t.replace("Fractional reduction in measured update time.", "Fractional reduction in propagation time with a precomputed closure.")
discussion = r'''\section{Discussion}
The results connect two decisions that are often evaluated separately: whether a graph prediction is correct enough to accept, and how its evidence can change the answer. \StableKG{} makes this connection inspectable. Calibration orders predictions by empirical correctness, the discrete certificate describes resistance to whole-window deletion under the additive scorer, and continuous diagnostics expose the concentration of support. Their roles are complementary at the protocol level: correctness ordering and sensitivity to evidence edits are different measurable outcomes.

The matched controls sharpen this interpretation. A normalized margin accounts for much of the selective gain, while the deletion diagnostic more accurately identifies susceptibility to window removal than raw margin. The strongest descriptor therefore depends on the decision being made: accepting a prediction and anticipating the effect of an evidence edit call for different summaries. Neural transfer and chronological replay connect this distinction to learned predictions and sequentially revealed histories. Dataset-specific operating behavior is visible in the full risk--coverage results.

Recent work on temporal evidence selection, graph editing and memory-based graph representations provides natural settings for such decision layers \citep{dao2025hgct,fan2026editing,jin2025memorywalk,liu2025terdy}. The TeRDy transfer and public-graph maintenance experiments show that the protocol remains useful as the predictive encoder and event volume change. The additive certificate supplies an exact local guarantee, while the indexed implementation provides an auditable route from event edits to refreshed decisions.

\section{Conclusion}
\StableKG{} connects calibrated acceptance with explicit evidence dependence in temporal knowledge graph reasoning. It improves selective decisions on two public ICEWS benchmarks, transfers to a learned temporal scorer, and retains its advantage during chronological replay. Direct interventions make the relation between support structure and answer stability measurable, while dependency closure provides an exact mechanism for maintaining derived conclusions efficiently. Together, these results establish an inspectable foundation for deciding which evolving graph predictions to use and when to revisit their supporting evidence.
'''
replace_between(r"\section{Discussion}", r"\section{Methods}", discussion)
certificate = r'''\subsection{Window-deletion certificate and continuous diagnostics}
Partition winner evidence into bins $k=\lfloor t_i/7\rfloor$, each spanning seven integer timestamp units. Let $v_1\geq\cdots\geq v_B>0$ be the sorted bin masses, $M=\sum_jv_j$, and $\Delta>0$ the winner margin with competitor evidence fixed. Define
\begin{equation}
 k_*(q)=\min\left\{k:\sum_{j=1}^{k}v_j\geq\Delta\right\},\qquad
 f(q)=\frac{\sum_{j=1}^{k_*}v_j}{M}.
\end{equation}
\textbf{Deletion guarantee.} Every deletion of fewer than $k_*$ whole winner windows leaves a strictly positive margin; some deletion of $k_*$ windows erases that margin. To see this, any subset of $j$ windows has total mass no greater than the $j$ largest masses. All prefixes shorter than $k_*$ sum to less than $\Delta$, and the $k_*$ prefix reaches it. This proves both statements. Equality erases the strict margin but can retain the original answer under deterministic tie-breaking.

The count $k_*$ is the discrete certificate. The associated $f$ records the mass cost of the descending-prefix attack. Zero evidence or a non-positive margin gives $k_*=f=0$. The implementation handles floating-point differences when a full-mass deletion reaches the margin. The continuous descriptors below serve calibration and intervention diagnosis.

Temporal diversity and the continuous deletion diagnostic are
\begin{equation}
 d(q)=1-\frac{v_1}{M},\qquad c(q)=\frac{f(q)+d(q)}{2},
\end{equation}
with $d=0$ when $M=0$. The normalized evidence margin is
\begin{equation}
 z(q)=\operatorname{clip}\left(\frac{\Delta}{M(\hat o)+M(o_{(2)})+10^{-12}},0,1\right).
\end{equation}
An added runner-up mass of $\Delta$ erases the strict margin when other masses are fixed; $z$ expresses that gap relative to the top-two mass. It is a calibration feature, distinct from the fixed one-event injection experiment. For neural candidates, these diagnostics use the selected candidate's evidence and strongest evidence competitor, clipping a negative evidence margin to zero. They describe the evidence scorer associated with that candidate.
'''
replace_between(r"\subsection{Stability certificate}", r"\subsection{Calibration", certificate)
t = t.replace("This causal cutoff is also used when constructing filtered answer sets.", "Known answers at the same query key are used only for filtered ranking, separately from the scoring evidence.")
t = t.replace("The calibration and threshold protocol is unchanged.", "The first half of validation timestamps fits the calibrator and the second half sets operating thresholds.")
replace_between("The update graph has raw-event nodes", r"\subsection{Data and implementation}", r'''The controlled update graph contains base-fact and derived-rule nodes. Reverse edges connect each premise to rules that read it. For a changed base set, traversal collects all reachable derived nodes; recomputation follows their topological order. Each rule aggregates premise values as one half their minimum plus one half their mean, and a conclusion takes the maximum over its rules. The runtime comparison perturbs 10\% of base nodes and times 2,000 propagation repetitions with the closure precomputed. Closure discovery is outside that timing. Node-count reduction is measured over the dynamic stream's changed-base sets, while runtime uses a separately sampled update set. Full and incremental outputs are compared elementwise.''')
t = t.replace("Their query format follows recent temporal-graph benchmark practice for explicit history prefixes and time-aware evaluation~\\citep{zhu2025multigran}; archive landing pages are linked in the manifest.", "The point-query format is supplied by TFLEX; its source repository and archive landing pages are linked in the manifest.")
t = t.replace("results\\_final\\_v4", "results\\_final\\_v5")
t = t.replace("the smoke tests and the exact Python dependencies are listed in \\texttt{requirements.txt}", "tests are in \\texttt{tests/}; dependency bounds and the tested environment are recorded in \\texttt{requirements.txt} and \\texttt{requirements-lock.txt}")
replace_between(r"\section{AI-use disclosure}", r"\section*{CRediT", r'''\section{AI-use disclosure}
OpenAI Codex assisted with research planning, software implementation, execution of experiments, statistical analysis, figure generation, literature metadata checking and manuscript drafting. Experimental values were produced by the accompanying executable code and linked to saved outputs. The author is responsible for reviewing the methods, data provenance, interpretation, references and final manuscript before submission.''')
for old, new in {
    "71.6\\%": f"{100*syn['node_reduction']['mean']:.1f}\\%",
    "47.7\\%": f"{100*syn['runtime_reduction']['mean']:.1f}\\%",
    "62.6 nodes": f"{syn['incremental_nodes']['mean']:.1f} nodes",
    "70.7--72.3\\%": f"{100*syn['node_reduction']['ci_low']:.1f}--{100*syn['node_reduction']['ci_high']:.1f}\\%",
    "42.2--52.8\\%": f"{100*syn['runtime_reduction']['ci_low']:.1f}--{100*syn['runtime_reduction']['ci_high']:.1f}\\%",
    "0.881 to 0.455 ms": f"{syn['full_update_ms']['mean']:.3f} to {syn['incremental_update_ms']['mean']:.3f} ms",
}.items():
    t = t.replace(old, new)
(PAPER / "main.tex").write_text(t, encoding="utf-8")

# The supplement is generated from source tables to prevent transcription drift.
si = r'''\documentclass[preprint,12pt]{elsarticle}
\usepackage{amsmath,amssymb,booktabs,longtable,graphicx,geometry,xurl}
\usepackage[hidelinks]{hyperref}
\geometry{margin=1in}
\renewcommand{\thetable}{S\arabic{table}}
\begin{document}
\begin{center}\Large\bfseries Supplementary Information\\[0.4em]
Stability certificates for selective reasoning over evolving knowledge graphs\end{center}
\section*{Evaluation record}
Public runs use 5,000 evenly sampled validation and test point queries, five validation permutations, a fixed ridge penalty of 0.01, half-life 35 timestamp units and bins of seven timestamp units. Neural results use three trained checkpoints. Chronological validation is split in time into calibration and threshold periods. All reported evidence diagnostics use the corrected full-mass deletion calculation. Original runs are preserved separately.
'''
def table(caption, headers, rows, align=None):
    align = align or "l" + "r"*(len(headers)-1)
    out = "\\begin{table}[htbp]\n\\centering\n\\caption{" + caption + "}\n\\small\n\\resizebox{\\linewidth}{!}{%\n\\begin{tabular}{" + align + "}\n\\toprule\n"
    out += " & ".join(headers) + "\\\\\n\\midrule\n"
    out += "\n".join(" & ".join(row) + r"\\" for row in rows)
    return out + "\n\\bottomrule\n\\end{tabular}}\n\\end{table}\n"
bs = lambda s, k: f"{s['baseline_'+k]:.4f}/{s['stability_'+k]:.4f}"
si += table("Public results. B/S denotes baseline/StableKG. Coverage targets are set on validation data; realized test coverage can differ.", ["Dataset", "MRR", "Brier B/S", "ECE B/S", "AURC B/S", "Risk20 B/S", "Risk40 B/S"], [[n, f"{pub[n]['decay_mrr']:.4f}"] + [bs(pub[n], k) for k in ("brier", "ece", "aurc", "risk@0.20", "risk@0.40")] for n in names])
rows = [[r.dataset, r.metric.replace("_", r"\_"), f"{r.estimate:.4f}", f"{r.ci_low:.4f}", f"{r.ci_high:.4f}"] for r in paired.itertuples()]
si += table("Paired differences (StableKG minus baseline), seed 0, 1,000 resamples of contiguous timestamp blocks. Risk uses frozen validation thresholds separately for each method; these are not equal accepted-count comparisons.", ["Dataset", "Metric", "Estimate", "2.5\\%", "97.5\\%"], rows, "llrrr")
labels = {"margin_only":"Margin only", "matched_base":"B", "base_plus_c":"B + $c$", "base_plus_z":"B + $z$", "base_plus_fd":"B + $f,d$", "without_diversity":"B + $z,f$", "without_fragility":"B + $z,d$", "nonredundant":"B + $z,f,d$", "full_original":"Full", "base_plus_window_count":"B + $k_*$", "base_plus_softmax_entropy":"B + softmax, entropy"}
si += table("Matched-control AURC over five validation permutations. B contains margin, log-support and recency. All configurations share the same calibration/operating splits and ridge penalty. The entire feature grid was retained.", ["Features", *names], [[label] + [f"{controls[n].loc[k,'aurc']:.4f}" for n in names] for k,label in labels.items()])
si += "The full feature model contains a redundant linear combination because $c=(f+d)/2$. Its inclusion affects ridge geometry, not the information available. Paired block-bootstrap contrasts for each control are included in the source CSV files. These comparisons are diagnostic; no model was selected using the test results.\n"
si += table("Intervention flip rates in low/high diagnostic-score quartiles. The window spans seven timestamp units. Queries with no winner evidence are excluded, and tied quartiles can contain different counts.", ["Dataset", "Queries", "Single event", "Window", "Counter-event"], [[n, str(inter[n]["n_queries"])] + [f"{inter[n]['low_certificate_'+e+'_flip']:.4f}/{inter[n]['high_certificate_'+e+'_flip']:.4f}" for e in ("event","window","counter")] for n in names])
si += table("Neural and chronological results. B/S denotes baseline/StableKG.", ["Protocol", "Dataset", "AURC B/S", "Risk20 B/S", "Risk40 B/S"], [[mode,n]+[bs(source[n],k) for k in ("aurc","risk@0.20","risk@0.40")] for mode,source in [("Neural",neural),("Chronological",chrono)] for n in names[:2]], "llrrr")
si += table("Recent author-code TeRDy comparison. Metrics are from the completed GPU runs under the shared Pe panel.", ["Dataset", "Rank", "Epochs", "MRR", "AURC B", "AURC S", "Risk20 B/S"], [["ICEWS14","6000","12","0.6571","0.3801","0.2896","0.1880/0.1530"],["ICEWS05-15","2000","8","0.5355","0.3801","0.3716","0.2330/0.2210"]], "lrrrrrr")
for n in names:
    d = updates[n]
    g = d[(d.kind != 'clock') & (d.batch == 100)]
    si += table(f"Public update maintenance for {n}. Means over 10 repetitions; all outputs matched the full refresh.", ["Events", "Decisions", "Batch", "Full ms", "Incremental ms", "Refreshed"], [[f"{int(d.events.iloc[0]):,}",f"{int(d.n_keys.iloc[0]):,}","100",f"{g.full_elapsed_ms.mean():.1f}",f"{g.incremental_elapsed_ms.mean():.1f}",f"{g.incremental_refreshed.mean():.1f}"]], "rrrrrr")
si += table("Controlled rule-DAG update study, 20 seeds. Runtime excludes closure discovery and measures propagation over a cached closure. Runtime and node counts use different changed-base samples.", ["Metric", "Mean", "2.5\\%", "97.5\\%"], [[label]+[f"{100*syn[key][stat]:.2f}\\%" for stat in ("mean","ci_low","ci_high")] for label,key in [("Node reduction","node_reduction"),("Propagation-time reduction","runtime_reduction")]] + [["Maximum output discrepancy",str(syn["incremental_max_error"]["mean"]),"0","0"]])
si += table("Intervention-survival AUROC for matched diagnostics. Higher is better. $c$ is the deletion diagnostic and $z$ the normalized margin; raw margin provides the control for a fixed injected event.", ["Dataset", "Edit", "$c$", "$z$", "Margin", "Support"], [[n,e]+[f"{diagnostics[(diagnostics.dataset==n)&(diagnostics.edit==e)&(diagnostics.feature==k)].iloc[0].survival_auroc:.4f}" for k in ("certificate","counter_evidence_cost","margin","support")] for n in names for e in ("event","window","counter")],"llrrrr")
si += table("Exactly equal accepted-count comparison, calibration seed 0. Both frozen score orderings accept the same number of test queries. Coverage budgets (20\\%, 40\\%) are prespecified; ties retain fixed source order. Intervals use 1,000 paired timestamp-block resamples and recompute each ordering at that budget. This evaluates a fixed budget, separately from the deployable validation-threshold policy.", ["Dataset", "Coverage", "Risk B", "Risk S", "Difference", "2.5\\%", "97.5\\%"], [[r.dataset,f"{100*r.coverage:.0f}\\%",f"{r.baseline_risk:.4f}",f"{r.stability_risk:.4f}",f"{r.delta_risk:.4f}",f"{r.ci_low:.4f}",f"{r.ci_high:.4f}"] for r in fixed.itertuples()])
si += r'''\clearpage
\section*{Reproduction and interpretation}
Run \texttt{run\_all.ps1 -RunSynthetic -RunNeural -CompilePaper} to regenerate the experiments, statistics, figures and manuscript. The local checkpoint refresh command is recorded in \texttt{scripts/refresh\_evidence.py}. Tests cover causal cutoffs, filtered ranking, actual predict-before-reveal replay, the discrete window-count guarantee, floating-point full-mass deletion and incremental/full agreement.

The public interpolation-style split permits training events at the query timestamp. Chronological replay instead predicts before revealing that timestamp. Selective correctness means that the unfiltered top prediction belongs to the published query answer set; filtered MRR separately removes other known same-query answers. Thus selective correctness and filtered Hits@1 have different evaluation denominators. Timestamp units are dataset indices and are not assumed to be calendar days. The additive-scorer certificate does not certify neural-parameter robustness.

The prefix attack minimizes the number of deleted windows, not their mass over arbitrary subsets. For masses $(6,4,3)$ and margin 7 it removes two windows with mass 10, whereas another two-window deletion has mass 7. This distinction fixes the interpretation of the guarantee without treating the continuous diagnostic as a universal robustness radius.

\section*{Source files}
Corrected public outputs are in \path{results_final_v5/}; interventions, neural inference and chronological replay have separate version-5 directories. Matched controls and raw feature caches are in \path{results_review_20260909/}. Each table here is generated from those files. Bibliographic metadata responses and retrieval errors are retained under \path{literature/verification_20260909/}.
\end{document}
'''
(PAPER / "supplementary.tex").write_text(si, encoding="utf-8")
print("Revised manuscript and source-derived supplementary tables written.")
