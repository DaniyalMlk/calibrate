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
  correlation,
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

export {
  simulateMatrix,
  simulateResponse,
  simulateResponses,
} from './simulation/respondent.js';

export {
  MISSING,
  parseResponseCsv,
  ResponseMatrix,
  type Cell,
  type ParseCsvOptions,
  type ResponseMatrixInit,
} from './calibration/matrix.js';

export {
  screenExtremes,
  type Exclusion,
  type ExclusionReason,
  type ScreenResult,
} from './calibration/screen.js';

export {
  calibrate,
  toItems,
  type CalibratedItem,
  type CalibrationModel,
  type CalibrationOptions,
  type CalibrationResult,
} from './calibration/jmle.js';

export {
  itemFit,
  standardiseMeanSquare,
  type FitOptions,
  type FitReport,
  type ItemFit,
} from './calibration/fit.js';

export {
  bankHealth,
  healthTable,
  type BankHealth,
  type BankHealthOptions,
  type CoverageGap,
  type CoveragePoint,
} from './calibration/health.js';

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
  exposureChiSquare,
  exposureRates,
  exposureRatesFromIds,
  exposureVariance,
  overlapRate,
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

export { ItemPool } from './session/pool.js';

export {
  allOf,
  anyOf,
  fixedLength,
  maximumItems,
  precisionTarget,
  standardErrorBelow,
  withMinimumLength,
  type StopReason,
  type StoppingContext,
  type StoppingRule,
} from './session/stopping.js';

export {
  eapEstimator,
  hybridEstimator,
  mapEstimator,
  mleEstimator,
  wleEstimator,
  type HybridEstimatorOptions,
  type SessionEstimator,
} from './session/estimator.js';

export {
  AdaptiveSession,
  type SessionConfig,
  type SessionSnapshot,
  type SessionStatus,
  type TranscriptEntry,
} from './session/session.js';

export { syntheticBank, syntheticPool, type SyntheticBankOptions } from './simulation/bank.js';

export {
  drawPopulation,
  evenGridPopulation,
  gridPopulation,
  normalPopulation,
  uniformPopulation,
  type AbilityDistribution,
} from './simulation/population.js';

export {
  compareStudies,
  runStudy,
  simulateSession,
  type Policy,
  type SessionOutcome,
  type StudyOptions,
  type StudyResult,
} from './simulation/study.js';

export {
  conditionalReport,
  NORMAL_95,
  type ConditionalBin,
  type ConditionalOptions,
} from './simulation/conditional.js';

export {
  comparisonTable,
  conditionalTable,
  exposureSummary,
  summarise,
  type ExposureSummary,
  type StudySummary,
  type SummaryOptions,
} from './simulation/report.js';

export {
  recoveryStudy,
  recoveryTable,
  type RecoveryOptions,
  type RecoveryPoint,
  type RecoveryStudy,
} from './simulation/recovery.js';
