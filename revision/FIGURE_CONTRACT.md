# Neurocomputing figure contract

Backend: Python/matplotlib, editable text in PDF and SVG. Main-text width
183 mm; type 7–9 pt, never below 5 pt. White background, restrained charcoal,
teal, copper and violet palette. Each empirical panel uses every prespecified
dataset/backbone combination; no outcome-dependent filtering. Query subsampling
for timing was specified before evaluation and is labelled separately.

1. **Architecture and counterexample** (schematic-led composite). The same
   scores can correspond to different deletion stability. A neural/memory
   interface, two explicitly conceptual histories, and an exhaustive toy
   response illustrate why a certificate is additional information.
2. **Predictive effect** (paired quantitative panels). Compare neural and
   hybrid MRR with seed variability, and show paired gains with timestamp-block
   intervals. This establishes the predictive operating regime.
3. **Certified selection** (quantitative grid). Correctness-risk and edit-failure
   curves share fixed predictions; certified coverage limits are visible. No
   extrapolation beyond the available certified pool.
4. **Perturbation diagnosis** (quantitative grid). Compare ordinary margin and
   certificate surplus for distinct interventions; show paired AUROC differences
   and conditional block-bootstrap intervals across all four groups.
5. **Exactness and trade-offs** (quantitative grid). Exact versus conservative
   coverage, random-probe false safety, and prespecified fusion-weight response.
6. **Local maintenance** (quantitative grid). Identical write transactions,
   affected-query workload, measured runtime and cached memory. GDELT is a
   frequency-anchor scaling experiment, labelled explicitly.

Source-data CSVs accompany all quantitative panels. Conceptual values are
labelled as such. Seed means and sample SD describe independent model fits;
paired 95% intervals describe block resampling conditional on those fixed fits.
All figures receive measured-geometry checks, final PDF text/collision checks,
and rendered visual inspection.

Raster previews are exported at 600 dpi; vector PDF/SVG are the publication
masters. Source preflight has two reviewed warnings: it does not evaluate the
`183/25.4` inch conversion and misreads that source expression as 183 inches;
the rendered PDF/layout manifest supplies the actual 183 mm measurement.
It also requests TIFF when PNG previews accompany vectors. The line plots use
editable PDF/SVG masters, so a separate TIFF is unnecessary.
