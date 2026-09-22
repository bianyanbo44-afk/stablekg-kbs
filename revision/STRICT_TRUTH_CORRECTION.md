# Split-truth correction before final manuscript release

The first complete revision evaluation used a globally complete known-answer
set for validation filtering and correctness. A final isolation check found
119 admissible answers outside the designated ICEWS14 validation answer lists
and 111 outside the ICEWS05-15 lists. This could expose test-only truths to
validation decisions. It was therefore unsuitable for the intended strict
validation-only protocol.

The corrected adapter constructs validation known-answer sets exclusively from
training events and the complete validation event pool. Final test evaluation
retains the complete published truth set for standard filtered ranking and
known-answer correctness. The upstream cache is unchanged. A regression test
ensures a test-only answer never enters validation labels or filtering.

All six TeRDy fits were retrained with the same seeds, optimization settings,
epoch budgets, ranks and checkpoint rule, now using the strict validation
filter. The archived compact fits were trained for a fixed 30 epochs and need
no model reselection. Their validation logits were reused. Every fusion and
selector choice was refitted on corrected validation labels using the original
prespecified grids. All 12 corrected tests were evaluated only after freezing
these choices. No test-driven parameter changes were introduced.

Superseded outputs are preserved locally in
`results_nc_superseded_global_truth` and excluded from the release. The first
set of draft numbers and figures was replaced by the strict outputs.
The final primary consistency check covers 60,000 query–seed rows and 180,000
budget checks, with no certificate violations or witness mismatches
(`revision/CORE_CHECKS.json`). This is a protocol correction after an exploratory
test evaluation, not a claim that the development team had never seen test
outcomes.
