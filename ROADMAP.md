# Roadmap

Phases are ordered by dependency, not by date. Each one is expected to land with
tests that exercise the numerics against closed-form or published results.

## Phase 1 — Item response models

- [x] Dichotomous response functions (Rasch/1PL, 2PL, 3PL, 4PL)
- [x] Fisher item information and test information
- [x] Standard error of measurement from test information
- [x] Numeric guards: extreme abilities, degenerate discriminations, guessing bounds
- [x] Closed-form validation suite

## Phase 2 — Ability estimation

- [x] Log-likelihood, score function and observed information for a response pattern
- [x] Maximum likelihood estimation (Newton–Raphson with bisection fallback)
- [x] Gauss–Hermite and fixed-grid quadrature
- [x] EAP and MAP estimators with posterior standard deviation
- [x] Boundary handling for all-correct and all-incorrect patterns
- [x] Warm's weighted likelihood estimator (bias-corrected, prior-free)
- [x] Recovery study: bias and RMSE across the ability range

## Phase 3 — Item selection and exposure control

- [x] Maximum Fisher information selection
- [x] Kullback–Leibler information selection
- [x] Randomesque (top-k random) exposure control
- [x] Sympson–Hetter exposure parameters
- [x] Content balancing against a blueprint
- [x] Exposure-rate and bank-coverage accounting

## Phase 4 — Adaptive session engine

- [x] Item pool with indexing and eligibility filters
- [x] Session state machine: administer, score, re-estimate, stop
- [x] Stopping rules: standard-error threshold, fixed length, min/max bounds
- [x] Pluggable estimators, including a hybrid Bayesian-to-likelihood switch
- [x] Deterministic, replayable session transcripts
- [x] Serialisable session snapshots

## Phase 5 — Simulation harness

- [x] Synthetic item banks with realistic parameter distributions
- [x] Simulated examinees drawn from a specified ability distribution
- [x] Test-overlap rate and exposure chi-square
- [x] Policy comparison: measurement precision vs. test length vs. pool exposure
- [x] Conditional standard error and item-exposure reporting
- [x] Command-line entry point

## Phase 6 — Item bank and calibration

- [ ] Response-matrix ingest
- [ ] Joint maximum likelihood calibration of item parameters
- [ ] Item fit statistics (infit/outfit mean squares)
- [ ] Bank health report: information coverage gaps across the ability range

## Phase 7 — Web interface

- [ ] Live adaptive session view
- [ ] Ability posterior and confidence band, updating per response
- [ ] Item and test information curves
- [ ] Pool coverage and exposure visualisation
