import matplotlib.pyplot as plt
from calibration import compute_pred_reliability, compute_uncertainty_reliability


def plot_pred_reliability(class_probs, y_true, bins=10, min_obs_per_bin=5):
    bin_centers, confidence, acc = compute_pred_reliability(class_probs, y_true, bins=bins, min_obs_per_bin=min_obs_per_bin)

    y = confidence - acc
    fig = plt.figure()
    plt.plot(bin_centers, y, marker='.', markersize=8)
    plt.axhline(0, xmin=0., xmax=1., color='k', ls='--')
    plt.xlim([0., 1.])
    plt.ylim([-1., 1.])
    plt.grid()
    plt.title('Reliability diagram')
    plt.xlabel('Predicted probability')
    plt.ylabel('Predicted probability - Accuracy')
    plt.close()

    return fig


def plot_uncertainty_reliability(class_probs, posterior_params, y_true, calibration_model=None, bins=10, min_obs_per_bin=5):
    exp_probs, obs_probs = compute_uncertainty_reliability(class_probs, posterior_params, y_true, calibration_model=calibration_model,
                                                           bins=bins, min_obs_per_bin=min_obs_per_bin)

    fig = plt.figure()
    plt.plot(exp_probs, obs_probs, color='C1', marker='.', markersize=8)
    plt.axline([0, 0], [1, 1], color='k', ls='--')
    plt.xlim([0., 1.])
    plt.ylim([0., 1.])
    plt.grid()
    plt.title('P-P plot')
    plt.xlabel('Empirical cdf')
    plt.ylabel('Fitted theoretical cdf')
    plt.close()

    return fig
