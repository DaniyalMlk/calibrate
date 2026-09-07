/**
 * calibrate — an adaptive testing engine built on item response theory.
 *
 * The public surface is deliberately flat: item construction, the response
 * model, and the information functions that everything adaptive is built on.
 */

export {
  bracketSignChange,
  safeguardedRoot,
  type RootOptions,
  type RootResult,
} from './core/root.js';

export { createRng, normalDeviate, type Rng } from './core/random.js';

export {
  gaussHermite,
  integrate,
  normalGaussHermiteRule,
  normalGridRule,
  type QuadratureRule,
} from './core/quadrature.js';

export {
  clamp,
  linspace,
  logistic,
  logisticDerivative,
  LOGISTIC_METRIC_SCALE,
  mean,
  NORMAL_METRIC_SCALE,
  requireFinite,
  rootMeanSquareError,
  sum,
  variance,
} from './core/numeric.js';

export {
  describeModel,
  fourPL,
  makeItem,
  metricScale,
  onePL,
  rasch,
  threePL,
  twoPL,
  validateItemParameters,
  type Item,
  type ItemParameters,
  type Metric,
} from './models/item.js';

export {
  expectedScore,
  exponent,
  informationPeak,
  itemInformation,
  probabilityCorrect,
  probabilityIncorrect,
  probabilityOfResponse,
  responseDerivative,
  responseSecondDerivative,
  standardError,
  testInformation,
  type Response,
  type ScoredResponse,
} from './models/response.js';

export {
  expectedInformation,
  logLikelihood,
  observedInformation,
  patternBoundedness,
  scoreFunction,
} from './estimation/likelihood.js';

export {
  estimateMle,
  type AbilityEstimate,
  type BoundaryFlag,
  type EstimatorName,
  type MleOptions,
} from './estimation/mle.js';

export {
  normalPrior,
  priorInformation,
  priorLogDensity,
  priorScore,
  STANDARD_NORMAL_PRIOR,
  type NormalPrior,
} from './estimation/prior.js';

export {
  estimateEap,
  estimateMap,
  type BayesOptions,
  type MapOptions,
  type PosteriorEstimate,
  type QuadratureKind,
} from './estimation/bayes.js';

export {
  estimateWle,
  warmCorrection,
  weightedScore,
  type WleOptions,
} from './estimation/wle.js';

export { simulateResponse, simulateResponses } from './simulation/respondent.js';

export { simpson } from './core/quadrature.js';

export {
  rankByScore,
  type SelectionContext,
  type Selector,
} from './selection/selector.js';

export {
  kullbackLeiblerDivergence,
  kullbackLeiblerSelector,
  maximumInformationSelector,
  type KullbackLeiblerOptions,
} from './selection/information.js';

export {
  exposureRates,
  randomesque,
  rankByInformationAt,
  sympsonHetter,
  unusedFraction,
  type ExposureParameters,
  type SympsonHetterOptions,
} from './selection/exposure.js';

export {
  blueprint,
  contentBalanced,
  contentProportions,
  type Blueprint,
} from './selection/content.js';
