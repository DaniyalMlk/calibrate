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

## Phase 13 — Model-based differential item functioning

- [x] Dense linear solver, matrix inverse and the weighted normal-equation products
- [x] Logistic regression by iteratively reweighted least squares
- [x] Nested likelihood-ratio tests for uniform, non-uniform and combined DIF
- [x] Nagelkerke R-squared and the change in it as an effect size
- [x] Both published effect-size cut-point sets, with the narrower as default
- [x] Convergence reported rather than assumed, for separated data
- [x] Raju's signed and unsigned area measures in closed form
- [x] The crossing point, and the share of the departure the signed area misses
- [x] Areas validated against numerical integration of the same curves
- [x] Per-group calibration and linking wired into the command

## Phase 14 — Marginal maximum likelihood calibration

- [x] Marginal likelihood of a response pattern, integrated on a quadrature rule
- [x] Expectation step: ability posteriors accumulated into expected item counts
- [x] Maximisation step: one weighted logistic fit per item, Rasch and 2PL
- [x] Convergence judged on the marginal log-likelihood, with monotonicity asserted
- [x] Item-only screening, keeping the perfect and zero scorers marginal estimation can use
- [x] Cross-product and expected-count standard errors for the item parameters
- [x] Population distribution estimated from the data, with its shape reported
- [x] Validation against the closed-form single-item case and against joint estimation
- [x] Command-line entry point comparing the two estimators on one set of responses

## Phase 15 — Reporting a fixed form

- [x] Distribution of total scores at an ability, by recursion over items
- [x] The same distribution marginalised over a population
- [x] Score likelihoods across the ability grid, computed once per form
- [x] Expected a posteriori scoring against an arbitrary population, not only a normal prior
- [x] Score conversion table: ability, standard error and population share per total
- [x] Marginal reliability, and the average standard error behind it
- [x] Sufficiency of the total score under Rasch, asserted over every response pattern
- [x] Both item formats, binary and rubric-scored, on one form
- [x] Scoring report wired into the calibration command

## Phase 16 — Distribution

- [x] Continuous integration: the suite, the type check and the build, on the
      Node versions `engines` claims
- [x] The built binary exercised as a user runs it, with the command list read
      from the program rather than from the workflow
- [x] A manifest that can be published: licence file, repository metadata,
      resolvable engines floor, both bin names, no source maps over absent sources
- [x] A build that clears its output directory, so a pack cannot ship artefacts
      from an earlier configuration
- [x] Packaging tests asserting the tarball's actual contents
- [x] Tag-driven release workflow publishing over OIDC, with no stored credential
- [x] An evidence table of what was checked and against what, recomputed by a
      gate CI runs and pinned to the document in both directions
- [ ] A published release on the registry

## Phase 17 — Person fit

- [x] Standardised log-likelihood of a response pattern, with the moments taken
      over categories so both item formats share one path
- [x] The correction for an ability estimated from the same responses, derived
      against the score equation the estimator solves
- [x] Infit and outfit mean squares for a person, standardised
- [x] A scan over a sitting, worst first, with the threshold stated
- [x] Departures planted on purpose: preknowledge, carelessness, guessing
- [x] Detection scored against the key, with the false positive rate beside it
- [x] Command-line entry point

## Phase 18 — A front page, generated from the engine

- [x] A generated site, not a written one: the generator returns strings and
      touches no filesystem, so the tests call it directly and one thin entry
      point writes `site/`
- [x] The interface on it, built at publish time and importing the engine from
      the same compiled source the package ships. No bundler is involved,
      because there is none to involve: `tsc` emits ES modules whose import
      specifiers the browser resolves, which is why this can be published from
      a static site at all
- [x] Every figure from `scripts/evidence.ts`, including the two headline ones,
      which are read back out of the row they summarise rather than typed
      beside it — a headline is the figure most likely to be written once and
      left
- [x] Counts read off the command table and the package's own export list
- [x] A test that fails in both directions, and a page that is refused if any
      row of the gate is unhappy, so a failing measurement cannot be published
      as a result
- [x] The suite size stated below and checked: exactly on the file count,
      within a quarter on the test count, because a suite cannot count itself
      from inside

1179 tests across 50 files, typecheck clean.

## Phase 19 — The metric an anchor was calibrated on

- [x] The moment methods compare the slope of the exponent rather than the
      discrimination, so an anchor whose two calibrations chose different
      metrics links correctly instead of reporting the convention as scale
- [x] A cross-metric anchor in the evidence gate, where the transformation is
      the identity and the other three methods are the reference
- [x] The identity rows that could not see the metric — peak information
      against its closed form — measured on both metrics
- [x] Raju's areas measured between two calibrations that chose differently,
      the closed form read in the slope of each exponent rather than once for
      the pair, against the quadrature that reads each item's own metric

## Phase 20 — Options nobody has chosen

- [x] A gate over the source, parsed rather than grepped: every optional
      property of an exported interface has to be assigned somewhere outside
      the file that declares it, so an option that only ever gets its default
      is reported rather than merely covered
- [x] The survey asserted before the claim is made, since "nothing is
      unexercised" passes trivially over an empty one
- [x] The five options that were unexercised chosen with an assertion that the
      choice changed something, not merely that it was accepted
- [x] No exemption mechanism, deliberately: an option nobody will choose once
      in a test is an option to delete

## Phase 21 — What the engine does not do

- [x] The known gaps written down in the README, one bullet each, saying what
      is absent and what it costs a reader who assumed otherwise
- [x] Every bullet paired with a predicate over the package's own surface, so
      that closing a gap breaks the test and the line has to come out
      deliberately rather than be left behind to be believed
- [x] The inventory built by parsing the source rather than grepping it: the
      prose explaining why something is absent contains the word for it, and a
      grep would read that as the feature arriving
- [x] The inventory asserted before any predicate reads it, since "nothing is
      named X" passes trivially over an empty one
- [x] Five ways to fail — a line without a predicate, a predicate without a
      line, a list that shrinks, a bullet reduced to its lead, and a predicate
      that stops holding — and every one of them verified by being broken
