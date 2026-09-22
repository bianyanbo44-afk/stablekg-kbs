StableKG provides exact shared-window deletion certificates for a frozen neural
predictor with an editable, renormalized temporal memory. This release accompanies
the manuscript **Exact stability certificates for neural temporal knowledge
graph completion**, prepared for submission to Neurocomputing.

Included assets provide the main manuscript, Supplementary Information, editable
LaTeX and vector figures, and the complete author submission bundle. Query-level
source data are supplied as the `neurocomputing_source_data.zip` release asset,
with a SHA-256 manifest tracked in the repository. The restore command downloads
and verifies that asset automatically. The release does not imply journal
acceptance or peer-review endorsement.

The revised experiments use two neural backbones, two public completion datasets,
three training seeds, exact and sampled interventions, and a separate GDELT
maintenance benchmark. Validation decisions follow the strict split-specific
truth protocol documented in the repository. Earlier KBS outputs remain historical
artifacts and are not evidence for the revised manuscript.

See `README_NEUROCOMPUTING.md` for the build-only route from archived predictions
and the complete training/evaluation route. Public dataset sources, terms and
hashes are in `data/DATA_MANIFEST.md`. Original code is MIT licensed; datasets and
external model implementations retain their own source terms.
