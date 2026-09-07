/**
 * calibrate — an adaptive testing engine built on item response theory.
 *
 * The public surface is deliberately flat: item construction, the response
 * model, and the information functions that everything adaptive is built on.
 */

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
  standardError,
  testInformation,
  type Response,
  type ScoredResponse,
} from './models/response.js';
