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
        # Consider both positive and negative class
        b_pos = np.abs(class_probs[i] - bin_centers).argmin()
        b_neg = np.abs(1. - class_probs[i] - bin_centers).argmin()
        confidence[b_pos] += class_probs[i]
        confidence[b_neg] += (1. - class_probs[i])
        if y_true[i] == 1:
            correct_count[b_pos] += 1
        elif y_true[i] == 0:
            correct_count[b_neg] += 1
        count[b_pos] += 1
        count[b_neg] += 1

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

    return bin_centers, confidence, acc


def compute_uncertainty_reliability(class_probs, posterior_params, y_true, calibration_model=None, bins=10, min_obs_per_bin=5):
    assert len(class_probs) == len(posterior_params), "class_probs and posterior_params must have same length."
    
    N = len(class_probs)
    bin_centers, _, acc = compute_pred_reliability(class_probs, y_true, bins=bins, min_obs_per_bin=min_obs_per_bin)

    cum_probs = np.zeros_like(class_probs)
    for i in range(N):
        b = np.abs(class_probs[i] - bin_centers).argmin()
        alpha, beta = posterior_params[i]
        cum_probs[i] = sp.stats.beta.cdf(acc[b], alpha, beta)
        if calibration_model is not None:
            if not np.isnan(cum_probs[i]):
                cum_probs[i] = calibration_model.predict([cum_probs[i]])[0]

    exp_probs = np.linspace(0.0, 1.0, N + 1)
    cum_probs = cum_probs[~np.isnan(cum_probs)] # Remove NaNs
    obs_probs = np.array([(cum_probs <= x).sum() / len(cum_probs) for x in exp_probs])

    return exp_probs, obs_probs


def fit_calibration_model(calibration_data):
    """Fit calibration model using isotonic regression."""
    exp_probs, obs_probs = calibration_data
    X, y = obs_probs, exp_probs

    calibration_model = IsotonicRegression(increasing=True, y_min=0.0, y_max=1.0)
    calibration_model = calibration_model.fit(X, y)
    
    return calibration_model
    