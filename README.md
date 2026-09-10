# calibrate

An adaptive testing engine built on item response theory.

A fixed-length quiz spends most of its questions telling you what it already
knows: easy items are wasted on strong candidates, hard ones on weak candidates.
An adaptive test picks each next question to maximise what it learns from the
answer, and stops when the estimate is precise enough. `calibrate` implements
that loop — the response models, the ability estimators, the item-selection
policies and the stopping rules — as a dependency-free TypeScript library.

## Status

Phases 1 to 9 of [ROADMAP.md](ROADMAP.md) are complete. An adaptive test runs
end to end — `npm run demo` administers one against a synthetic bank and prints
the transcript — `npm run study` compares selection policies over a simulated
population, `npm run fit` estimates item parameters from a matrix of responses,
and `npm run web` serves a browser interface that runs a session live and draws
the posterior, the information curves and the bank's coverage as it goes.

Items may be scored in two categories or in many. The graded response, partial
credit and generalized partial credit models sit alongside the dichotomous ones,
and a single bank can hold both: `npm run mixed` runs an adaptive session over
one. The web interface is still dichotomous-only, which is phase 11.

## Install and run

```bash
npm install
npm test          # vitest
npm run typecheck # tsc --noEmit, library and web interface, including tests
npm run build     # emits dist/ with declarations

npm run demo      # one adaptive session, with its transcript
npm run mixed     # an adaptive session over a bank mixing both item formats
npm run study     # selection policies compared over a simulated population
npm run recover   # how well each estimator recovers a known ability
npm run fit       # item parameters calibrated from a matrix of responses
npm run web       # the browser interface, on http://localhost:5173/web/
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

### Items scored in more than two categories

Not every item is right or wrong. An essay marked against a four-level rubric, a
Likert item, a multi-step task where reaching step three is worth more than
reaching step one — all of these carry information in *how far* the response got,
and scoring them 0/1 throws most of it away.

Three polytomous models cover those formats. An item with `m` thresholds scores
into `m + 1` ordered categories, `0..m`:

```ts
import {
  categoryProbabilities,
  expectedCategoryScore,
  generalizedPartialCredit,
  graded,
  partialCredit,
  polytomousInformation,
} from 'calibrate';

const essay = graded(1.2, [-1.5, -0.2, 0.9]); // four rubric levels
categoryProbabilities(essay, 0.4); // [0.093, 0.235, 0.318, 0.354] — sums to one
expectedCategoryScore(essay, 0.4); // 1.934 out of 3
polytomousInformation(essay, 0.4); // 0.437
```

- `graded(a, thresholds)` — Samejima's graded response model. Thresholds are
  *cumulative* boundaries: `thresholds[k - 1]` is the ability at which reaching
  category `k` or higher becomes more likely than not. Use it when the rubric
  levels are ordered bands of one underlying quality.
- `partialCredit(thresholds)` — Masters' partial credit model, discrimination
  fixed at 1. Thresholds are *adjacent steps*: the ability at which categories
  `k - 1` and `k` are equally likely. Use it when the task is solved in sequence
  and each step is its own hurdle.
- `generalizedPartialCredit(a, thresholds)` — Muraki's version of the same, with
  a free discrimination per item.

The distinction changes whether thresholds must be ordered. Under the graded
model they must: unordered cumulative boundaries imply a negative category
probability, which is arithmetic rather than a modelling choice, so it is
rejected at construction. Under the partial credit models a reversal is
meaningful — it says the category is never the single most likely outcome at any
ability, which is a real finding about a narrow rubric band — so it is allowed
through.

Both families reduce to the dichotomous models when given a single threshold: a
two-category graded item *is* the 2PL, and a two-category partial credit item
*is* the Rasch model, agreeing to fourteen digits. The test suite checks this
against `models/response.ts` rather than against a second copy of the formula.

### Forms that mix both

A real test form usually mixes formats: forty multiple-choice items and two
extended responses. The mixed layer defines information, expected score and the
test characteristic curve once for items of either kind.

```ts
import {
  formatCounts,
  makeItem,
  makePolytomousItem,
  maximumTestScore,
  testCharacteristicCurve,
  testInformationOf,
  twoPL,
} from 'calibrate';

const form = [
  makeItem('mc-1', twoPL(1.4, 0.2)),
  makeItem('mc-2', twoPL(1.1, -0.3)),
  makePolytomousItem('essay-1', graded(1.2, [-1.5, -0.2, 0.9])),
];

formatCounts(form); // { dichotomous: 2, polytomous: 1, maximumScore: 5 }
testInformationOf(form, 0); // 1.219 — information from the whole form
testCharacteristicCurve(form, 0); // 2.684 — expected total score at that ability
maximumTestScore(form); // 5
```

The two formats are told apart by the shape of their parameters — a difficulty
versus a threshold vector — rather than by a tag, so an item cannot be built
with a label that contradicts its own parameters. A dichotomous item is handled
as the two-category case throughout: its category distribution is `[1 - P, P]`
and its expected score is `P`.

The test characteristic curve is the bridge between the ability metric and a
reported raw score. It rises monotonically to `maximumTestScore` — though not
necessarily from zero, since a 3PL item on the form contributes its lower
asymptote at any ability.

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

### Choosing the next item

```ts
import {
  blueprint,
  contentBalanced,
  maximumInformationSelector,
  randomesque,
} from 'calibrate';

const selector = contentBalanced(
  randomesque(maximumInformationSelector(), 5),
  blueprint({ arrays: 0.4, graphs: 0.3, dp: 0.3 }),
);

selector.select({ candidates, theta, responses, rng }); // Item | null
```

Policies share one interface and compose by wrapping, so the trade-off between
measurement precision and bank security is configuration rather than a rewrite.

- `maximumInformationSelector` — the textbook rule: the item that would tell you
  most about the ability you currently believe the candidate has.
- `kullbackLeiblerSelector` — integrates over a window around the current
  estimate instead of committing to a point. Early in a test the estimate could
  be off by a logit, and an item that separates the whole plausible interval is
  worth more than one that is razor-sharp at a number nobody believes yet.
- `randomesque` — pick uniformly among the top `k`.
- `sympsonHetter` — administer a selected item only with probability `K_i`,
  falling through to the next candidate on rejection.
- `contentBalanced` — restrict candidates to the domain furthest behind its
  blueprint target, then apply the base rule inside it.

Exposure control is not a nicety. An unconstrained maximum-information rule
administers the same handful of high-discrimination items to nearly every
candidate; those items leak, and a leaked bank is worth less than it cost to
build. `exposureRates` and `unusedFraction` are there so the effect of a policy
can be measured rather than asserted — in the test suite, `randomesque` is
verified to lower both the peak exposure rate and the fraction of the bank left
untouched.

### Running a test

```bash
npm run demo -- --theta 0.8 --target 0.3 --max 30
```

```
#   item                            b      a  score    theta      se  est
----------------------------------------------------------------------------
1   arrays-0114                 -0.31   1.50  right    0.283   0.937  eap
2   dynamic-programming-0233    -0.37   1.69  right    0.506   0.865  eap
...
9   graphs-0136                  1.44   1.75  wrong    1.697   0.646  wle
...
30  arrays-0069                  0.14   1.54  right    1.110   0.323  wle

Finished after 30 items: reached the 30-item ceiling
Estimate 1.110 (true 0.800, error +0.310, 0.96 standard errors)
```

The `est` column is worth watching: while every answer is correct there is no
finite likelihood estimate, so a Bayesian estimator carries the ability. At item
9 the candidate gets one wrong, the pattern becomes mixed, and the session
switches to the weighted likelihood estimator for the rest of the test.

In code:

```ts
import {
  AdaptiveSession,
  ItemPool,
  maximumInformationSelector,
  precisionTarget,
} from 'calibrate';

const session = new AdaptiveSession({
  pool: new ItemPool(items),
  selector: maximumInformationSelector(),
  stopping: precisionTarget(0.3, { minimum: 5, maximum: 30 }),
  seed: 42,
});

while (session.status !== 'finished') {
  const item = session.nextItem();
  if (item === null) break;
  session.submit(await ask(item)); // 0 or 1
}

session.snapshot(); // theta, standardError, stopReason, full transcript
```

Stopping rules are composable values — `fixedLength`, `standardErrorBelow`,
`maximumItems`, `withMinimumLength`, `anyOf`, `allOf` — and `precisionTarget` is
the conventional combination of a floor, a target and a ceiling.

### Answering an item in more than two ways

Scoring is done in categories throughout: the log-likelihood, the score
function, the observed information and Warm's bias correction each sum over the
category actually scored rather than over a right/wrong flag. A dichotomous
response is the two-category case of that, not a separate path — selecting
category `u` from `[1 - P, P]` picks out exactly the term `u log P + (1 - u) log Q`
keeps — so every dichotomous result is unchanged.

```ts
import { estimateEap, estimateMle, estimateWle } from 'calibrate';

const pattern = [
  { item: mc1, response: 1 },           // a binary item
  { item: mc2, response: 0 },
  { item: essay, category: 2 },         // 2 of 3 on the rubric
  { item: task, category: 1 },
];

estimateMle(pattern); // one likelihood, both formats
estimateWle(pattern);
estimateEap(pattern);
```

Boundedness is where the mixed case genuinely differs. A pattern has no finite
maximum likelihood estimate only when every item scored *its own* maximum, which
on a mixed form is not the same as "all correct": a candidate who answers every
multiple-choice item correctly and scores 2 of 3 on the essay has a perfectly
finite estimate, because the middle category says where on the scale they sit.
Comparing each response against its own item's maximum rather than against 1 is
the whole of the difference.

Selection widens the same way. Kullback-Leibler divergence sums over all
categories, so a four-category item separates two abilities through four
probability ratios instead of one. The two-term dichotomous form is the sum over
`[Q, P]`, so no dichotomous selection decision changes.

### Running a mixed-format test

```
$ npm run mixed -- --theta 0.6 --seed 4242

Bank: 300 items — 225 binary, 75 scored in more than two categories
Candidate: true ability 0.60
Stopping: standard error <= 0.3, between 5 and 25 items

#   item                          format    loc     a  score    theta      se  est
-------------------------------------------------------------------------------------
1   dynamic-programming-cr-0203    3 cat   0.59  2.08    0/2   -0.624   0.787  eap
2   arrays-cr-0075                 4 cat   0.40  2.43    1/3   -0.568   0.672  wle
3   arrays-cr-0231                 4 cat  -0.03  1.76    1/3   -0.576   0.565  wle
4   graphs-cr-0235                 4 cat  -0.55  1.96    3/3   -0.172   0.470  wle
5   graphs-cr-0139                 5 cat   1.03  1.71    1/4   -0.180   0.431  wle
...
13  dynamic-programming-cr-0107    4 cat   1.96  1.67    0/3    0.343   0.298  wle

Finished after 13 items (12 of them scored in more than two categories)
Raw score 20 of 41 points; the model expected 19.13 at this ability
Estimate 0.343 (true 0.600, error -0.257, 0.86 standard errors)
Standard error 0.298
Item maxima ranged from 1 to 4 points, so 13 items were worth 41 points.
```

Two things in that transcript are worth noticing. The selection policy reaches
for the rubric items first — 12 of 13 — because a four-category item carries
several times the information of a binary one at the same location, which is the
practical argument for mixing formats at all. And the test stopped after 13
items but 41 *points*: on a mixed form those are different numbers, and
conflating them is the usual way a mixed test gets misreported.

### Comparing policies

One session tells you nothing about a policy. `npm run study` runs a whole
simulated population through several of them against the same bank and the same
examinees, and prints what they cost each other:

```
policy                  items    bias    rmse      se   calib   cover      r   max_x  overlap  unused
-----------------------------------------------------------------------------------------------------
max-information          17.1  -0.021   0.430   0.411    0.95   0.965  0.927   1.000    0.389   0.643
kullback-leibler         17.1  -0.024   0.432   0.410    0.95   0.965  0.927   1.000    0.391   0.643
randomesque-5            18.1  -0.035   0.445   0.404    0.91   0.930  0.923   0.755    0.355   0.610
balanced+randomesque     20.1  -0.018   0.457   0.417    0.91   0.940  0.923   0.585    0.290   0.490
fixed-30                 30.0  -0.023   0.346   0.333    0.96   0.965  0.954   1.000    0.426   0.543
```

Two things fall out of that table. The adaptive policies reach a standard error
of 0.4 in about 17 items where the fixed form takes 30 — and the fixed form buys
its extra precision (RMSE 0.35 against 0.43) with those thirteen extra items,
not with anything cleverer. And exposure control is nearly free: content
balancing on top of randomesque cuts the peak exposure rate from 1.00 to 0.59
and the overlap between two candidates' tests from 0.39 to 0.29, for three extra
items and no measurable loss of precision.

The columns worth knowing:

- **calib** — the standard error the test reported, divided by the error it
  actually made. One is honest; below one means the test is promising a
  precision it does not have, which is the failure that matters when a cut score
  is applied to the result.
- **cover** — the fraction of candidates whose true ability fell inside their
  reported 95% interval.
- **overlap** — the expected proportion of items two randomly chosen candidates
  saw in common, `(N/L)S² + L/N` for a bank of `N` items and tests of length
  `L`. It ranges from `L/N` under perfectly even exposure to exactly 1 when
  everyone sits the same form, and it is the number to watch for bank security:
  a bank can have a respectable peak exposure rate and still be trivially
  harvestable if the items travel together.

Marginal numbers hide the thing a bank owner most needs to see, so every study
also reports conditionally on true ability:

```
max-information — conditional on true ability
ability              n    bias    rmse      se   calib   cover   items
----------------------------------------------------------------------
[-3.0, -2.0)         5  -0.212   1.436   0.992    0.69   0.800    25.8
[-2.0, -1.0)        27   0.122   0.346   0.397    1.15   0.963    21.9
[-1.0, 0.0)         67  -0.005   0.398   0.395    0.99   0.970    15.1
[0.0, 1.0)          61  -0.092   0.349   0.394    1.13   0.967    14.7
[1.0, 2.0)          31   0.005   0.311   0.394    1.27   1.000    18.9
[2.0, 3.0)           9  -0.076   0.519   0.413    0.80   0.889    24.0
  107 of 300 items used, 29 above an exposure rate of 0.2, overlap 0.389
```

The shape of that column of test lengths is the bank's own shape: candidates
between -1 and 1 reach the target in about 15 items, and candidates below -2
take 26 and still finish with a standard error of 0.99 and a calibration of
0.69. The bank is thin down there, and the marginal RMSE of 0.43 says nothing
about it.

Binning is on the *generating* ability, never on the estimate. Conditioning on
the estimate is a classic way to make a biased test look unbiased: the
candidates a test overestimates are exactly the ones it moves into a higher bin,
so the error gets smuggled into the conditioning variable and disappears.

### Checking the estimators

`npm run recover` answers the prior question — whether the estimators recover an
ability that is known, before any of them is trusted to estimate one that is
not. It simulates a fixed form many times at each of a set of abilities:

```
eap — mean absolute bias 0.6593, pooled RMSE 0.9384
   theta     n     mean    bias    rmse      se   calib   cover   bound
-----------------------------------------------------------------------
   -3.00   200   -1.430   1.570   1.612   0.616    0.38   0.175   0.000
   -2.00   200   -1.216   0.784   0.887   0.601    0.68   0.825   0.000
   -1.00   200   -0.774   0.226   0.481   0.581    1.21   0.980   0.000
    0.00   200   -0.002  -0.002   0.467   0.552    1.18   0.990   0.000
    1.00   200    0.757  -0.243   0.522   0.553    1.06   0.950   0.000
    2.00   200    1.376  -0.624   0.748   0.581    0.78   0.860   0.000
    3.00   200    1.835  -1.165   1.224   0.621    0.51   0.570   0.000
```

That is shrinkage, stated numerically. A candidate at 3.0 is reported at 1.84 on
average, and only 57% of them have their true ability inside their own 95%
interval, because the prior is pulling every estimate towards the population it
assumes they came from. On the same form Warm's estimator has a mean absolute
bias of 0.14 against EAP's 0.66 and MAP's 0.71. The test suite asserts these
relationships rather than the numbers: shrinkage towards the prior mean, growing
with distance from it; Warm's estimator less than half as biased as either
Bayesian estimator; maximum likelihood leaving 29% of the patterns at `theta =
-3` with no interior estimate at all, where the others resolve every one.

The `bound` column is why the tails of the maximum likelihood table are not
directly comparable to the others. Patterns with no interior maximum are
excluded from the row rather than folded in at the edge of the search window;
including them would make an estimator's apparent bias depend on how wide its
window happens to be. The fraction excluded is reported instead, which carries
the same information without pretending a bound is an estimate.

### Calibrating a bank

Everything above assumes the item parameters are known. They are not: a bank
starts as a matrix of who answered what, and the parameters have to be estimated
from it.

```bash
npm run fit -- --bank 12 --respondents 800 --target 0.9
```

```
Source: simulated: 800 respondents, 12 rasch items
Model: rasch, converged in 8 iterations (largest change 2.06e-5)
Screened: 10 respondents and 0 items removed over 2 passes; 790 respondents and 12 items calibrated

item                     p+       b   se(b)      a   infit  outfit   t(in)   flag
-----------------------------------------------------------------------------------
arrays-0000           0.759  -1.977   0.092   1.00   0.969   0.948   -0.66
graphs-0001           0.533  -0.730   0.083   1.00   1.009   1.039    0.27
dynamic-programming-0002  0.390  -0.013   0.085   1.00   0.952   0.966   -1.31
arrays-0003           0.242   0.832   0.094   1.00   0.918   0.887   -1.75
graphs-0004           0.085   2.245   0.131   1.00   0.908   0.772   -1.03
...
```

In code:

```ts
import { calibrate, itemFit, parseResponseCsv, toItems } from 'calibrate';

const matrix = parseResponseCsv(csv, { index: true });
const result = calibrate(matrix, { model: 'rasch' });
const fit = itemFit(matrix, result);

result.items; // id, difficulty, standard error, proportion correct
fit.misfitting; // the items that do not behave the way the model says
toItems(result); // ready to hand to an ItemPool
```

Estimation alternates: hold the item parameters fixed and estimate every
ability, hold the abilities fixed and estimate every item, repeat until neither
moves. Each half is a well-behaved one- or two-parameter problem even though the
joint problem over thousands of parameters is not. On a simulated 200-item Rasch
bank answered by 800 people it converges in six iterations and recovers the
generating difficulties with a correlation of 0.997 and a mean absolute error of
0.078.

Three details do most of the work:

**Extreme respondents and items have to go first, and removal cascades.** A
person who answered everything correctly has no finite ability estimate, only a
lower bound; an item everybody passed has the same problem. Joint estimation
with either present does not fail loudly, it walks off towards infinity dragging
the scale with it. And screening cannot be a single sweep: dropping the strongest
candidates can leave an item nobody remaining answered correctly, which then has
to go too, which can make another person extreme. `screenExtremes` loops until a
pass removes nothing and reports what went, why, and on which pass.

**The scale has to be pinned every cycle.** Adding a constant to every ability
and every difficulty leaves the likelihood exactly unchanged, and under the 2PL
so does stretching the ability scale while shrinking the discriminations to
match. Left alone the estimates slide along those directions forever and the
convergence test never fires — not because the fit is improving but because the
parameters are moving along a ridge. Under Rasch the mean difficulty is fixed at
zero; under the 2PL, where both the location and the unit are free, the
abilities are standardised to mean zero and unit variance instead.

**The estimates are biased, and the bias has a known correction.** Because the
number of person parameters grows with the sample, the usual consistency
argument does not apply and joint difficulty estimates come out inflated away
from zero. Multiplying them by `(L - 1) / L` for a test of `L` items is the
classical fix; on the bank above it takes the mean absolute error from 0.068 to
0.060, and at 150 respondents from 0.163 to 0.152. That is a Rasch result, so it
is applied by default only there — under the 2PL the discriminations absorb part
of the same bias and the correction measurably overshoots.

### Finding items that do not fit

A calibrated parameter is not the same as a working item. Infit and outfit mean
squares compare each item's observed responses against what the model predicted:
both are means of squared standardised residuals, differing in whether the
residuals are weighted by their own variance. Infit weights them, which damps
the responses of candidates far from the item's difficulty — where a surprise is
cheap — and emphasises those near it, where the item is doing its work. Outfit
does not, which makes it the more sensitive of the two to a single very
unexpected answer.

In a suite test, thirty Rasch items are simulated to fit, one is answered at
random, and one has its key reversed. The two planted items are the only two
flagged; the miskeyed one comes out at an outfit of 2.17 against the random
item's 1.38, because reversing a key does not add noise so much as invert the
relationship between ability and success.

The standardised deviates use the Wilson–Hilferty cube-root transform, because a
mean square is a ratio of chi-squares and strongly right-skewed: 1.3 is
unremarkable on thirty responses and damning on three thousand, and only the
standardised form says which situation you are in.

### How concentrated exposure is

The peak exposure rate, the overlap rate and the unused fraction each say
something true about bank security, and none of them describes the shape of the
distribution.

```ts
import { exposureConcentration, exposureRatesFromIds } from 'calibrate';

const rates = exposureRatesFromIds(sessions.map((session) => session.itemIds));
const shape = exposureConcentration(bank.length, rates);

shape.gini; // 0.704
shape.topDecileShare; // 0.42 — the most-exposed tenth carried 42% of everything
shape.curve; // the Lorenz curve, for plotting
```

A bank where thirty items carry sixty percent of all administrations and one
where a hundred and twenty items carry the same sixty percent can report an
identical peak rate and an identical unused fraction. Only the shape separates
them, and the shape is what says how much of the bank an attacker would have to
harvest to reconstruct most of the testing. The Gini coefficient is zero under
perfectly even use and `(n - 1) / n` when one item absorbs everything — not
one, because a bank of finite size cannot concentrate perfectly.

`bankSize` is a required argument rather than something inferred from the map
because an item that was never administered is the most concentrating thing a
bank can hold, and a curve computed only over the items that were used would
omit exactly that.

### Where a bank cannot measure

```ts
import { bankHealth } from 'calibrate';

const health = bankHealth(items, { target: 0.3 });
health.gaps; // [{ from: 1.75, to: 3.00, worstStandardError: 0.48 }]
```

The question before writing more items is not how many you have but where the
next ones should sit. `bankHealth` walks the ability range, reports the test
information and standard error the whole bank could reach at each point, and
collapses the failures into intervals. Reporting "nothing above 1.75" is an
item-writing brief; twelve consecutive failing rows of a table is a puzzle.

The standard error is computed as though every item in the bank were
administered, which is deliberately optimistic — no candidate sees the whole
bank. That is what makes it a sufficient screen: a region the full bank cannot
measure to target is one no adaptive test over that bank will ever measure to
target.

### The posterior, not just its mean

`estimateEap` returns the posterior mean and `estimateMap` its mode, but a
session view wants the whole distribution: the shape of what is believed, and
how much of it lies where.

```ts
import { credibleInterval, posteriorDensity, posteriorMassBetween } from 'calibrate';

const posterior = posteriorDensity(responses); // even grid, normalised
posterior.mean; // 0.927
posterior.sd; // 0.301
posterior.mode; // interpolated, not the highest grid point

credibleInterval(posterior, 0.95); // { lower: 0.32, upper: 1.48, width: 1.16 }
posteriorMassBetween(posterior, 0.5, 4); // 0.93 — confidence in a pass at a cut of 0.5
```

The grid is evenly spaced, which is the whole reason this exists separately
from `estimateEap`. Gauss-Hermite nodes cluster near the prior mean and are the
right choice for computing a mean, but a band drawn between them has vertices
that bunch in the middle and stretch at the tails, and the eye reads that as
structure in the posterior rather than in the quadrature.

Intervals are equal-tailed rather than highest-density, for a reportability
reason rather than a numerical one. An equal-tailed interval is a pair of
posterior quantiles, so "2.5% of the posterior lies below this candidate's
interval" is a sentence that survives being quoted in a score report. A
highest-density interval is narrower on a skewed posterior, but its endpoints
are not quantiles of anything and it can be disjoint on a bimodal posterior —
which a mixed pattern on high-guessing items really can produce.

`edgeRatio` reports how much density is still standing at the edge of the grid,
as a fraction of the peak. Everything else in the result is conditioned on the
posterior lying inside the grid, and that is the number which says whether it
does.

## The web interface

```bash
npm run web
```

An adaptive session in the browser: answer items as correct or incorrect, or
let a simulated candidate at a chosen ability answer them, and watch the
posterior tighten per response. The selection policy, the target standard
error, the length ceiling and the simulated ability are all live controls, and
every panel re-renders from the same configuration.

Four panels:

- **Session** — the item awaiting a response and its parameters, the running
  estimate, and the stopping rule's verdict.
- **Ability posterior** — the density with its nested 50, 80 and 95 percent
  credible bands, redrawn per response.
- **Information and precision** — test information across the ability range,
  with a rug of item difficulties along the baseline, and the standard error of
  measurement below it on the same ability axis.
- **Bank coverage and exposure** — the standard error the whole bank could
  reach at each ability, with its coverage gaps, and a Lorenz curve over item
  exposure across a simulated population.

Three decisions shaped it.

**No bundler and no runtime dependencies.** `tsc` emits ES modules whose import
specifiers are already what a browser resolves, so the browser loads the
library directly — the same files a consumer would import, exercised the way a
consumer would exercise them, with nothing in between that could paper over a
mistake. Compiling with `rootDir` at the repository root preserves the relative
path from the interface to `src`, so one specifier works both in the
TypeScript source and in the emitted JavaScript. The charts are hand-written
SVG for the same reason: a linear scale, an axis that lands on round numbers
and a band between two edges are a few lines each, which is less than the cost
of a chart library on a page whose point is that it has none.

**The headline figure is the estimate the engine reports.** The posterior mean
sits beside it as its own labelled readout rather than in the headline. The two
disagree slightly by construction — one is the weighted likelihood estimate,
the other the mean of the posterior — and a headline showing one while the
transcript showed the other reads as an arithmetic error rather than as two
estimators.

**No panel has two vertical axes.** Test information and the standard error of
measurement are the obvious candidate: they are two views of one quantity and
they fit neatly on one plot with an axis on each side. But they are
reciprocally related, so the point where the two lines cross is set entirely by
the two ranges chosen and says nothing about the test. They are stacked on one
shared ability axis instead, which keeps every comparison horizontal — the only
direction in which those two quantities are comparable.

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

**Two threshold conventions, not one.** It would be tidier to store polytomous
thresholds in a single canonical form and convert. But a graded boundary and a
partial credit step are different quantities that happen to be the same shape:
one is cumulative, one is adjacent, and only the first has to be ordered.
Collapsing them would either impose an ordering constraint the partial credit
models do not have — discarding the diagnostically useful case of a reversed
step — or drop the constraint the graded model cannot do without. They stay
distinct, and `validatePolytomousParameters` applies the rule that belongs to
each.

**Discriminating on parameter shape, not on a tag field.** A mixed bank has to
tell its two item formats apart. A `kind: 'polytomous'` field would be
representable in a state that contradicts the parameters beside it. Testing for
the presence of `thresholds` cannot: the discriminator is the thing that
actually differs.

**One likelihood over categories, not two code paths.** The alternative was to
keep the dichotomous likelihood and add a parallel polytomous one. That would
have meant every estimator, every stopping rule and every selection policy
existing twice, and the two drifting. Rewriting the likelihood over category
indices instead makes the dichotomous case fall out as the two-category
special case. The evidence that this changed nothing is that the whole existing
suite — most of it dichotomous — passes untouched against the generalised code.

**One simulation draw convention.** `simulateCategory` delegates to
`simulateResponse` for dichotomous items rather than deriving them through its
own inverse-CDF loop. Both consume one uniform draw and both give the same
distribution, but they map a given draw to opposite outcomes: walking `[Q, P]`
upward returns 1 when `draw >= Q`, while the Bernoulli comparison returns 1 when
`draw < P`. Two conventions in one engine would mean a seeded simulation gave
different answers depending on which function the caller reached for, and every
replayable transcript would quietly change meaning.

**Bracket first, then Newton.** The 3PL log-likelihood is not guaranteed to be
unimodal, and its observed information can be negative — a correct answer to a
hard, high-guessing item sits on a convex stretch of the surface. A raw Newton
step there walks uphill in the wrong direction and leaves the ability range
entirely. The estimator scans for a sign change in the score function, then runs
Newton steps confined to that bracket, falling back to bisection whenever a step
would escape it. Slightly slower on easy problems; it cannot diverge.

**Content balancing is a hard restriction, not a penalty term.** The tempting
alternative is to subtract a content penalty from the information score, which
produces tests that are *nearly* balanced. "Nearly" is not a claim a testing
programme can put in a report: a test that promises 40% data structures has to
deliver 40% data structures. Restricting the candidate set to the domain
furthest behind target, then letting the psychometric rule choose inside it,
makes the blueprint an invariant rather than a preference.

**Every source of randomness is a seeded generator passed in.** Selection
policies receive an `Rng` in their context and are forbidden `Math.random`. A
policy whose behaviour cannot be replayed cannot be compared against another
one, and a session that cannot be replayed cannot be audited after a candidate
disputes their score.

**The session's status is enforced, not advisory.** Calling `submit` before
`nextItem`, or `nextItem` twice in a row, throws. A session that quietly accepts
a response to an item it never administered produces a transcript that cannot be
trusted, and the only reason to keep a transcript is that it can be.

**Every examinee in a study is seeded from their index, not from a shared
stream.** The obvious implementation threads one generator through the loop,
which makes examinee `i` depend on how much randomness the `i - 1` before them
consumed — and that, in turn, depends on the policy under test. A comparison
built that way is not a controlled comparison, because the examinees differ
between the arms. Deriving each examinee's seed from their index makes them
identical across policies and identical whether the study runs ten of them or
ten thousand, which is asserted in the test suite rather than assumed.

**Boundary patterns are excluded from a recovery row, not clamped into it.** An
all-correct pattern has no maximum likelihood estimate; reporting the edge of the
search window as though it were one would make the estimator's measured bias a
function of how wide that window was configured to be. The row reports the
fraction excluded instead, which says the same thing without inventing a number.

**Recovery is judged on centred parameters, not raw ones.** Joint estimation
identifies the parameters only up to the origin of the scale, and the calibrator
picks that origin by convention. A generating bank has no reason to share it, so
comparing raw numbers reports the difference between two conventions as though
it were estimation error — on a short bank that difference dominates everything
else, and it is not an error at all.

**Tests check against closed forms, not against previous output.** `P(b) = 0.5`
for the 2PL, `(1 + c) / 2` for the 3PL, an information peak of `a^2 / 4` at
`theta = b`, and a 3PL peak at `b + ln[(1 + sqrt(1 + 8c)) / 2] / (D a)` are all
derivable by hand. Snapshotting the implementation's own numbers would lock in
whatever it does today, including its bugs.

## License

MIT
