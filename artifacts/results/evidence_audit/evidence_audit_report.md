# Frozen-Artifact Evidence Audit

This audit performs no retraining or test-set selection. A successful audit run does not by itself make every readable artifact current evidence.

## Current UQ evidence

- Status: `CURRENT_ACCEPTED_MATCHED_HETEROSCEDASTIC`.
- Source: `artifacts/results/matched_heteroscedastic_uq`; its machine-readable acceptance audit passed.
- The referenced comparison uses the same grouped splits, heteroscedastic Gaussian likelihood, and total optimizer-update budget.
- `matched_heteroscedastic_uq_acceptance_reference.json` pins the validated source files and hashes.

## Legacy likelihood-mismatched UQ replay

- Status: `LEGACY_LIKELIHOOD_MISMATCHED`; `current_evidence_eligible=false`.
- `legacy_likelihood_mismatched_uq_proper_scores_by_seed.csv` and `legacy_likelihood_mismatched_uq_proper_scores_summary.csv` recompute scores from the older one-channel/global-noise dropout artifacts for archival reproducibility only.
- `uq_proper_scores_by_seed.csv` and `uq_proper_scores_summary.csv` remain readable compatibility aliases and carry the same non-propagating status columns.

## Historical retired proxy source

- Status: `HISTORICAL_RETIRED_NON_PROPAGATING`; `current_evidence_eligible=false`.
The archived proxy is a degree-3 polynomial with 84 normalized monomials. It is retained only as a legacy regression artifact and provides no validation for the replacement terminal-Q source.

The canonical historical filenames are `historical_retired_proxy_charge_coefficients.csv` and `historical_retired_proxy_charge_definition.json`. The old filenames remain compatibility aliases with identical explicit retirement metadata.

## Scope

The evidence matrix distinguishes completed numerical audits from unexecuted external simulator checks and unavailable independent measurement/process domains. Only rows with `current_evidence_eligible=true` may propagate into current-evidence claims.
