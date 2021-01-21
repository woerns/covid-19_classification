import numpy as np
import scipy as sp
import scipy.stats

import matplotlib.pyplot as plt


def plot_pred_reliability(class_probs, y_true, bins=10, min_obs_per_bin=5):
    assert len(class_probs) == len(y_true), "class_probs and y_true must have same length."

    bin_edges = np.linspace(0.0, 1.0, bins+1)
    bin_centers = np.array([(bin_edges[i]+bin_edges[i+1])/2 for i in range(len(bin_edges)-1)])
    correct_count = np.zeros_like(bin_centers)
    count = np.zeros_like(bin_centers)
    confidence = np.zeros_like(bin_centers)
    for i in range(len(class_probs)):
        # Consider both positive and negative class
        b_pos = np.abs(class_probs[i]-bin_centers).argmin()
        b_neg = np.abs(1.-class_probs[i]-bin_centers).argmin()
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


def plot_uncertainty_reliability(class_probs, posterior_params, y_true, bins=10, min_obs_per_bin=5):
    assert len(class_probs) == len(posterior_params) == len(y_true), "class_probs, posterior_params and y_true must have same length."

    N = len(class_probs)
    bin_edges = np.linspace(0.0, 1.0, bins+1)
    bin_centers = np.array([(bin_edges[i]+bin_edges[i+1])/2 for i in range(len(bin_edges)-1)])
    correct_count = np.zeros_like(bin_centers)
    count = np.zeros_like(bin_centers)
    for i in range(len(class_probs)):
        # Consider both positive and negative class
        b_pos = np.abs(class_probs[i]-bin_centers).argmin()
        b_neg = np.abs(1.-class_probs[i]-bin_centers).argmin()
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
        else:
            acc[b] = correct_count[b] / count[b]
            # Apply "rule of three" if none or all observations are positive
            significance_level = 0.1
            thresh = np.exp(np.log(significance_level) / count[b])
            acc[b] = np.clip(acc[b], a_min=1. - thresh, a_max=thresh)
    
    cum_probs = np.zeros_like(class_probs)
    for i in range(N):
        b = np.abs(class_probs[i] - bin_centers).argmin()
        alpha, beta = posterior_params[i]
        cum_probs[i] = sp.stats.beta.cdf(acc[b], alpha, beta)

    exp_probs = np.linspace(0.0, 1.0, N+1)
    y = np.array([(cum_probs<=x).sum()/(~np.isnan(cum_probs)).sum() for x in exp_probs])

    fig = plt.figure()
    plt.plot(exp_probs, y, color='C1', marker='.', markersize=8)
    plt.axline([0, 0], [1, 1], color='k', ls='--')
    plt.xlim([0., 1.])
    plt.ylim([0., 1.])
    plt.grid()
    plt.title('P-P plot')
    plt.xlabel('Empirical cdf')
    plt.ylabel('Fitted theoretical cdf')
    plt.close()

    return fig
