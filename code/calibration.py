import numpy as np
import scipy as sp

from sklearn.isotonic import IsotonicRegression


def compute_pred_reliability(class_probs, y_true, bins=10, min_obs_per_bin=5):
    assert len(class_probs) == len(y_true), "class_probs and y_true must have same length."

    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_centers = np.array([(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)])
    correct_count = np.zeros_like(bin_centers)
    count = np.zeros_like(bin_centers)
    confidence = np.zeros_like(bin_centers)
    for i in range(len(class_probs)):
        # Aggregate confidence across all classes
        for j in range(len(class_probs[i])):
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


def compute_expected_calibration_error(class_probs, y_true, bins=10, min_obs_per_bin=5):
    _, count, confidence, acc = compute_pred_reliability(class_probs, y_true, bins=bins, min_obs_per_bin=min_obs_per_bin)

    ece = np.nansum(np.abs(confidence - acc)*count)/count.sum()

    return ece


def compute_uncertainty_reliability(class_probs, posterior_params, y_true, calibration_model=None, bins=10, min_obs_per_bin=5):
    assert len(class_probs) == len(posterior_params), "class_probs and posterior_params must have same length."

    # TODO: Generalize to multi-class probabilities using Dirichlet distribution
    # Currently we are only considering max mean class prob for uncertainty estimates and their reliability
    # so we have to cast it into a binary classification problem (i.e. either max mean prob class or not)
    y_true = (class_probs.argmax(axis=-1) == y_true).astype(int) # pseudo labels
    class_probs = class_probs.max(axis=-1)
    class_probs = np.stack((1.-class_probs, class_probs), axis=1)

    N = len(class_probs)
    bin_centers, _, _, acc = compute_pred_reliability(class_probs, y_true, bins=bins, min_obs_per_bin=min_obs_per_bin)

    # We get two observed quantiles: One for predicted probability p and one for 1-p
    obs_probs = np.zeros(2*N,)
    for i in range(N):
        b_neg = np.abs(class_probs[i][0] - bin_centers).argmin()
        b_pos = np.abs(class_probs[i][1] - bin_centers).argmin()

        alpha, beta = posterior_params[i]
        obs_probs[2*i] = sp.stats.beta.cdf(acc[b_neg], beta, alpha)  # NOTE: alpha and beta are switched for opposite class.
        obs_probs[2*i + 1] = sp.stats.beta.cdf(acc[b_pos], alpha, beta)

    if calibration_model is not None:
        for j in range(2*N):
            if not np.isnan(obs_probs[j]):
                obs_probs[j] = calibration_model.predict([obs_probs[j]])[0]

    exp_cdf = np.linspace(0.0, 1.0, 2*N + 1)
    obs_probs = obs_probs[~np.isnan(obs_probs)] # Remove NaNs
    obs_cdf = np.array([(obs_probs <= x).sum() / len(obs_probs) for x in exp_cdf])

    return exp_cdf, obs_cdf


def compute_wasserstein_dist(cdf_x, cdf_y):
    """Computes Wasserstein distance for two given CDFs."""
    bin_edges = np.concatenate(([0.0], 0.5*(cdf_x[:-1] + cdf_x[1:]), [1.0]), axis=0)
    dx = (bin_edges[1:] - bin_edges[:-1])
    wasserstein_dist = (np.abs(cdf_x-cdf_y) * dx).sum()

    return wasserstein_dist


def compute_posterior_wasserstein_dist(class_probs, posterior_params, y_true, calibration_model=None, bins=10, min_obs_per_bin=5):
    exp_cdf, obs_cdf = compute_uncertainty_reliability(class_probs, posterior_params, y_true,
                                                           calibration_model=calibration_model,
                                                           bins=bins, min_obs_per_bin=min_obs_per_bin)
    return compute_wasserstein_dist(exp_cdf, obs_cdf)


def fit_calibration_model(calibration_data):
    """Fit calibration model using isotonic regression."""
    exp_probs, obs_probs = calibration_data
    X, y = obs_probs, exp_probs

    calibration_model = IsotonicRegression(increasing=True, y_min=0.0, y_max=1.0)
    calibration_model = calibration_model.fit(X, y)
    
    return calibration_model
    