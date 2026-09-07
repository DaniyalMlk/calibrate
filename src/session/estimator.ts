import { estimateEap, estimateMap, type BayesOptions, type MapOptions } from '../estimation/bayes.js';
import { patternBoundedness } from '../estimation/likelihood.js';
import { estimateMle, type AbilityEstimate, type MleOptions } from '../estimation/mle.js';
import { estimateWle, type WleOptions } from '../estimation/wle.js';
import { STANDARD_NORMAL_PRIOR } from '../estimation/prior.js';
import type { ScoredResponse } from '../models/response.js';

/**
 * How a session turns a response pattern into an ability estimate.
 *
 * A plain function rather than an interface, because that is all a session needs
 * and it lets callers supply their own without implementing anything.
 */
export type SessionEstimator = (responses: readonly ScoredResponse[]) => AbilityEstimate;

/** Expected a posteriori estimation. Defined for every pattern, including empty. */
export function eapEstimator(options: BayesOptions = {}): SessionEstimator {
  return (responses) => estimateEap(responses, options);
}

/** Maximum a posteriori estimation. */
export function mapEstimator(options: MapOptions = {}): SessionEstimator {
  return (responses) =>
    responses.length === 0 ? estimateEap(responses, options) : estimateMap(responses, options);
}

/** Maximum likelihood estimation, with an EAP fallback for the empty pattern. */
export function mleEstimator(options: MleOptions = {}): SessionEstimator {
  return (responses) =>
    responses.length === 0 ? estimateEap([]) : estimateMle(responses, options);
}

/** Warm's weighted likelihood estimation, with an EAP fallback for the empty pattern. */
export function wleEstimator(options: WleOptions = {}): SessionEstimator {
  return (responses) => (responses.length === 0 ? estimateEap([]) : estimateWle(responses, options));
}

export interface HybridEstimatorOptions {
  /** Options for the Bayesian phase. */
  readonly bayes?: BayesOptions;
  /** Options for the likelihood phase. */
  readonly likelihood?: WleOptions;
}

/**
 * The configuration operational adaptive tests actually run.
 *
 * While the pattern is all-correct or all-incorrect there is no finite
 * likelihood estimate, so a prior has to carry the estimate — otherwise the
 * first few items would be selected against an ability of plus or minus
 * infinity, which is to say the hardest or easiest item in the bank, every time.
 * Once the pattern is mixed the prior has done its job and shrinkage towards the
 * population mean becomes a liability, so the estimator switches to Warm's
 * weighted likelihood: bias-corrected, and answerable to the candidate rather
 * than to the population they were assumed to come from.
 *
 * Making the switch a first-class option rather than leaving it to callers means
 * the transcript records which estimator produced each entry, via the `method`
 * field on the estimate.
 */
export function hybridEstimator(options: HybridEstimatorOptions = {}): SessionEstimator {
  const bayes = options.bayes ?? { prior: STANDARD_NORMAL_PRIOR };
  const likelihood = options.likelihood ?? {};
  return (responses) => {
    const boundedness = patternBoundedness(responses);
    if (boundedness === 'bounded') return estimateWle(responses, likelihood);
    return estimateEap(responses, bayes);
  };
}
