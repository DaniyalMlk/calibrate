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

- [x] Response-matrix ingest, from arrays or CSV, with explicit missing responses
- [x] Iterative screening of extreme respondents and items
- [x] Joint maximum likelihood calibration of item parameters (Rasch and 2PL)
- [x] Scale identification and the joint-estimation bias correction
- [x] Item fit statistics (infit/outfit mean squares and standardised deviates)
- [x] Bank health report: information coverage gaps across the ability range

## Phase 7 — Web interface

- [x] Live adaptive session view
- [x] Ability posterior and credible band, updating per response
- [x] Item and test information curves
- [x] Pool coverage and exposure visualisation

## Phase 8 — Polytomous response models

- [x] Graded response model with ordered cumulative boundaries
- [x] Partial credit and generalized partial credit models
- [x] Category derivatives, first and second, with respect to ability
- [x] Polytomous item information and the partial credit variance form
- [x] Expected category score and the test characteristic curve
- [x] Mixed-format banks: one form holding both item formats
- [x] Numeric guards: softmax overflow, boundary cancellation, tail behaviour

## Phase 9 — Mixed-format measurement

- [x] Log-likelihood, score and observed information over categorical responses
- [x] Warm's bias correction summed over categories
- [x] Ability estimation from mixed-format response patterns
- [x] Boundedness of a mixed pattern: extreme scores across item maxima
- [x] Item selection and exposure accounting over a mixed pool
- [x] Adaptive sessions administering both formats
- [x] Synthetic banks mixing both formats
- [x] Command-line demonstration of a mixed-format test

## Phase 10 — Scale linking

- [x] The scale transformation, applied to items of either format
- [x] Mean/mean and mean/sigma moment methods
- [x] Haebara and Stocking-Lord characteristic curve methods
- [x] Derivative-free optimiser for the criterion methods
- [x] Common-item matching across two calibrations
- [x] Linking coefficients applied to a bank, both formats
- [x] Validation by exact recovery of a known transformation

## Phase 11 — Polytomous items in the web interface

- [x] Category response curves for a selected item
- [x] Rubric-scored responses in the live session view
- [x] Expected score and the test characteristic curve
- [x] Format mix in the bank coverage view
- [x] Levels that are modal nowhere, reported in the interface and the CLI

## Phase 12 — Differential item functioning

- [x] The regularized incomplete gamma function and the chi-square and normal tails
- [x] Stratification by a matching criterion, with thin strata merged
- [x] Mantel-Haenszel common odds ratio, chi-square and the ETS delta metric
- [x] The A/B/C classification, including the one-sided test against the boundary
- [x] Standardized proportion difference as a second effect size
- [x] Mantel's statistic and the standardized mean difference for rubric items
- [x] Two-group simulation with impact and bias controlled separately
- [x] Bank-wide scan, with unanalysable items reported rather than dropped
- [x] Iterative purification of the matching criterion
- [x] Command-line entry point
