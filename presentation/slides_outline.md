# Africa Growth Explorer: concise presentation outline

1. Question: can this indicator set predict next-year GDP-per-capita growth in the frozen African panel?
2. Protocol: train-only preprocessing, temporal outer folds, validation gate against global mean/persistence, frozen test window.
3. Outcome: test MAE 1.82; global-mean MAE 1.90. The descriptive row-bootstrap interval crosses zero; useful improvement is not established. Directional accuracy matches the majority always-up rule.
4. Failure analysis: country/year dependence, random internal early stopping, missing raw-data vintage and large country-year errors constrain interpretation.
5. Contribution: AI-assisted implementation, human-directed framing/review; no causal scenario or policy-forecast claim.

See reports/capstone_report.md and reports/generated/ for artifact-grounded values. Existing PDF exports are historical snapshots.
