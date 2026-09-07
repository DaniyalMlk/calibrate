# calibrate

An adaptive testing engine built on item response theory.

A fixed-length quiz spends most of its questions telling you what it already
knows: easy items are wasted on strong candidates, hard ones on weak candidates.
An adaptive test picks each next question to maximise what it learns from the
answer, and stops when the estimate is precise enough. `calibrate` implements
that loop — the response models, the ability estimators, the item-selection
policies and the stopping rules — as a dependency-free TypeScript library.

## Status

Phase 1 of [ROADMAP.md](ROADMAP.md) is complete — the item model, the response
functions and the information functions everything else is built on — and phase
2 is complete apart from its recovery study: four ability estimators, quadrature,
and the priors they run on.

## Install and run

```bash
npm install
npm test          # vitest
npm run typecheck # tsc --noEmit, including the test files
npm run build     # emits dist/ with declarations
```

## What is here today

```ts
import {
  itemInformation,
  makeItem,
  probabilityCorrect,
  standardError,
  testInformation,
  threePL,
  twoPL,
} from 'calibrate';

const item = twoPL(1.5, 0.5); // discrimination 1.5, difficulty 0.5
probabilityCorrect(item, 1.0); // 0.6792 — a candidate one logit up
itemInformation(item, 0.5); // 0.5625 — a^2 / 4, the peak of this item

const bank = [
  makeItem('arrays-1', twoPL(1.2, -1)),
  makeItem('graphs-4', threePL(1.4, 1.0, 0.25)), // multiple choice, four options
];
standardError(testInformation(bank, 0)); // how precise the estimate would be
```

### Item models

All four dichotomous models share one parameterisation, the four-parameter
logistic:

```
P(theta) = c + (d - c) / (1 + exp(-D * a * (theta - b)))
```

- `a` — discrimination, the slope at the inflection point
- `b` — difficulty, the ability at which the curve reaches its midpoint
- `c` — lower asymptote: what a candidate who knows nothing scores by guessing
- `d` — upper asymptote: one minus the probability that an expert slips
- `D` — the metric scaling, 1 for the logistic metric and 1.702 for the normal one

The Rasch/1PL, 2PL and 3PL are this same curve with parameters pinned, so
`rasch`, `onePL`, `twoPL`, `threePL` and `fourPL` all return the same shape and
the engine never branches on model type.

### Estimating ability

```ts
import { createRng, estimateMle, makeItem, simulateResponses, twoPL } from 'calibrate';

const bank = [-2, -1, 0, 1, 2].map((b, i) => makeItem(`i-${i}`, twoPL(1.2, b)));
const responses = simulateResponses(bank, 0.8, createRng(42));

const estimate = estimateMle(responses);
// { theta, standardError, method: 'mle', converged, boundary, iterations }
```

`boundary` is the field that matters. An all-correct or all-incorrect pattern
has a log-likelihood that is monotone in ability, so it has no finite maximiser
— the honest answer is "at least this high", not a number. The estimator returns
`boundary: 'upper'` or `'lower'` with `converged: false` and puts the edge of the
search window in `theta`, so a caller who wants a usable number has one but is
never told a bound is an estimate.

### Which estimator to use

Four are implemented, and the differences between them matter most exactly where
an adaptive test spends its early items: short patterns, extreme candidates.

| Estimator | Finite for all-correct? | Uses a prior | Where it fits |
|---|---|---|---|
| `estimateMle` | no | no | long tests, research |
| `estimateEap` | yes | yes | the opening items, where nothing else is defined |
| `estimateMap` | yes | yes | same, when the posterior mode is wanted instead of its mean |
| `estimateWle` | yes | no | reporting a final score |

```ts
import { estimateEap, estimateMap, estimateWle, normalPrior } from 'calibrate';

estimateEap(responses); // posterior mean, with posteriorSd
estimateMap(responses, { prior: normalPrior(0, 1.2) }); // posterior mode
estimateWle(responses); // bias-corrected, no prior
```

Maximum likelihood is biased outward: on a short test its estimates in the tails
are systematically too extreme. EAP and MAP fix that by shrinking towards the
prior mean, at the cost of assuming the candidate was drawn from that prior —
which is hard to justify to the individual whose score it lowered. Warm's
weighted likelihood estimator removes the same first-order bias without a prior,
by weighting the likelihood so that its asymmetry cancels. For a single Rasch
item answered correctly it returns `b + ln 3`, where maximum likelihood returns
infinity and EAP returns something that depends on the prior you picked.

### Information

The Fisher information an item carries at a given ability is

```
I(theta) = [P'(theta)]^2 / (P(theta) * (1 - P(theta)))
```

which is what makes adaptive selection possible: it says how much a candidate's
answer to this item would tell you about *this* candidate. Test information is
the sum over administered items, and its reciprocal square root is the standard
error of the ability estimate.

## Design decisions

**One parameterisation, not four.** The obvious alternative is a discriminated
union with a case per model and a `switch` in every function. Collapsing all
four onto the 4PL means the response function, its derivative and the
information function each exist once. The cost is that an invalid combination is
representable, which is why `validateItemParameters` runs at construction and
`describeModel` can report which model a parameter set really is.

**The general information formula, not the per-model shortcuts.** Textbooks give
`D^2 a^2 P Q` for the 2PL and the Birnbaum form for the 3PL. Both are algebraic
consequences of `[P']^2 / (PQ)`, so the engine computes the general form and the
test suite asserts that the shortcuts agree with it — the shortcuts become
checks rather than code paths.

**Failing loudly on the parameters that break estimation.** A discrimination
above about 20 turns the response curve into a step function within float
precision; the likelihood then has no usable gradient and every estimator
silently stops converging. Rejecting that at construction, with the offending
field named, is far cheaper to diagnose than a session that quietly returns
garbage.

**Bracket first, then Newton.** The 3PL log-likelihood is not guaranteed to be
unimodal, and its observed information can be negative — a correct answer to a
hard, high-guessing item sits on a convex stretch of the surface. A raw Newton
step there walks uphill in the wrong direction and leaves the ability range
entirely. The estimator scans for a sign change in the score function, then runs
Newton steps confined to that bracket, falling back to bisection whenever a step
would escape it. Slightly slower on easy problems; it cannot diverge.

**Tests check against closed forms, not against previous output.** `P(b) = 0.5`
for the 2PL, `(1 + c) / 2` for the 3PL, an information peak of `a^2 / 4` at
`theta = b`, and a 3PL peak at `b + ln[(1 + sqrt(1 + 8c)) / 2] / (D a)` are all
derivable by hand. Snapshotting the implementation's own numbers would lock in
whatever it does today, including its bugs.

## License

MIT
