import numpy as np
import scipy as sp

from sklearn.isotonic import IsotonicRegression


def compute_pred_reliability(class_probs, y_true, bins=20, min_obs_per_bin=10):
    assert len(class_probs) == len(y_true), "class_probs and y_true must have same length."

    # Split all probs uniformly into bins. Note that the bins can have different widths.
    # We do that because after enough training, all predictions tend to be either close to 0 or 1
    # and we need more granular buckets to estimate the true accuracy.
    # Alternatively, one could create fixed bins but with finer spacing (e.g. log spacing) at the edge of [0,1].
    N, C = class_probs.shape
    n_pad = ((N * C) // bins) * bins - N * C
    all_probs = class_probs.flatten()
    all_probs.sort()
    nans = np.empty((n_pad,))
    nans[:] = np.nan
    all_probs = np.append(all_probs, nans)
    bin_centers = all_probs.reshape((bins, -1)).mean(axis=1)

    correct_count = np.zeros_like(bin_centers)
    count = np.zeros_like(bin_centers)
    confidence = np.zeros_like(bin_centers)

    for i in range(N):
        # Aggregate confidence across all classes
        for j in range(C):
            # Find nearest bucket
            b = np.abs(class_probs[i][j] - bin_centers).argmin()
            confidence[b] += class_probs[i][j]

            if y_true[i] == j:
                correct_count[b] += 1

            count[b] += 1

    acc = np.zeros_like(bin_centers)
    for b in range(bins):
        if count[b] <= min_obs_per_bin:
            # Remove accuracy estimates with too few observations
            acc[b] = np.nan
            confidence[b] = np.nan
        else:
            acc[b] = correct_count[b] / count[b]
            # Apply "rule of three" if none or all observations are positive
            significance_level = 0.1
            thresh = np.exp(np.log(significance_level) / count[b])
            acc[b] = np.clip(acc[b], a_min=1. - thresh, a_max=thresh)

            confidence[b] = confidence[b] / count[b]

    return bin_centers, count, confidence, acc


def compute_expected_calibration_error(class_probs, y_true, bins=20, min_obs_per_bin=10):
    _, count, confidence, acc = compute_pred_reliability(class_probs, y_true, bins=bins, min_obs_per_bin=min_obs_per_bin)

    ece = np.nansum(np.abs(confidence - acc)*count)/count.sum()

    return ece


def compute_uncertainty_reliability(class_probs, posterior_params, y_true, calibration_model=None, bins=20, min_obs_per_bin=10):
    assert len(class_probs) == len(posterior_params[0]) == len(posterior_params[1]), "class_probs and posterior_params must have same length."

    bin_centers, _, _, acc = compute_pred_reliability(class_probs, y_true, bins=bins, min_obs_per_bin=min_obs_per_bin)

    N, C = class_probs.shape
    alpha, beta = posterior_params
    obs_probs = np.zeros((N, C))

    for i in range(N):
        for j in range(C):
            # Since the Beta distribution is generally not symmetric but is mirrored when its parameters
            # alpha and beta are exchanged, we map all distributions to be left-tailed (where alpha > beta) before
            # measuring and calibrating uncertainty quantiles.
            b = np.abs(class_probs[i][j] - bin_centers).argmin()
            if alpha[i][j] < beta[i][j]:
                obs_probs[i][j] = sp.stats.beta.cdf(1. - acc[b], beta[i][j], alpha[i][j])
            else:
                obs_probs[i][j] = sp.stats.beta.cdf(acc[b], alpha[i][j], beta[i][j])

    obs_probs = obs_probs.flatten()
    obs_probs = obs_probs[~np.isnan(obs_probs)]  # Remove NaNs
    eps = 1e-10  # Note: Too small numbers cause problems with calibration model
    obs_probs[obs_probs < eps] = 0.0

    if calibration_model is not None:
        obs_probs_calibrated = calibration_model.predict(obs_probs)
        if calibration_model.out_of_bounds == 'nan':
            # Use uncalibrated values if observed probability outside of training data range.
            obs_probs = np.where(np.isnan(obs_probs_calibrated), obs_probs, obs_probs_calibrated)
        else:
            obs_probs = obs_probs_calibrated

    obs_probs.sort()  # sort in ascending order
    exp_cdf = obs_probs
    obs_cdf = np.array([(obs_probs <= x).sum() / len(obs_probs) for x in exp_cdf])

    return exp_cdf, obs_cdf


def compute_wasserstein_dist(cdf_x, cdf_y):
    """Computes Wasserstein distance for two given CDFs."""
    bin_edges = np.concatenate(([0.0], 0.5*(cdf_x[:-1] + cdf_x[1:]), [1.0]), axis=0)
    dx = (bin_edges[1:] - bin_edges[:-1])
    wasserstein_dist = (np.abs(cdf_x-cdf_y) * dx).sum()

    return wasserstein_dist


def compute_posterior_wasserstein_dist(class_probs, posterior_params, y_true, calibration_model=None, bins=20, min_obs_per_bin=10):
    exp_cdf, obs_cdf = compute_uncertainty_reliability(class_probs, posterior_params, y_true,
                                                           calibration_model=calibration_model,
                                                           bins=bins, min_obs_per_bin=min_obs_per_bin)
    return compute_wasserstein_dist(exp_cdf, obs_cdf)


def fit_calibration_model(calibration_data):
    """Fit calibration model using isotonic regression."""
    exp_cdf, obs_cdf = calibration_data
    X, y = exp_cdf, obs_cdf

    calibration_model = IsotonicRegression(increasing=True, y_min=0.0, y_max=1.0, out_of_bounds='clip')
    calibration_model = calibration_model.fit(X, y)
    
    return calibration_model
    