import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import matplotlib.pyplot as plt
import os

#1D92cb

def errorbar_plot(values, labels, out_file, limits=None, Tc_line=None):
    markers = ['s', 'H', 'd', 'v', 'p']
    colors = ['#2A9Df8', '#FF920B', '#65e41d', '#be67ff', '#ff7e79']
    markeredgecolors = ['#0256a3', '#ed4c18',  '#00B000', '#6633cc',  '#ee2324']
    x_values = values['x']
    y_values = values['y']
    assert x_values.shape == y_values.shape, ('x and y data have different'
                                              + ' shapes.')
    y_err = values['y_err']
    fig_labels = labels['fig_labels']
    x_label = labels['x_label']
    y_label = labels['y_label']
    x_lim = limits.get('x_lim')
    y_lim = limits.get('y_lim')
    num_graphs = len(fig_labels)
    
    fig, ax = plt.subplots()
    if Tc_line is not None:
        ax.axvline(x=Tc_line, linestyle='--', color='k')
    for i in range(num_graphs):
        ax.errorbar(x_values[i], y_values[i], yerr=y_err[i], label=fig_labels[i],
                    marker=markers[i], markersize=5, fillstyle='full',
                    color=colors[i], markeredgecolor=markeredgecolors[i],
                    ls='-', lw=2., elinewidth=2., capsize=2., capthick=2.)
    leg = ax.legend(loc='best', markerscale=1.5, fontsize=14)
    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    if x_lim is not None:
        ax.set_xlim(x_lim[0], x_lim[1])
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])
    
    fig.tight_layout()
    print(f"Saving file to: {out_file}")
    fig.savefig(out_file, dpi=400, bbox_inches='tight')
    return fig, ax

def get_plot_num(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    existing = [
        int(i.split('_')[-1].rstrip('.png')) for i in os.listdir(out_dir) if
        i.endswith('.png')
    ]
    try:
        latest_num = max(existing)
    except ValueError:
        latest_num = 0
    return latest_num + 1



def block_resampling(data, num_blocks):
    """ Block-resample data to return num_blocks samples of original data. """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    num_samples = data.shape[0]
    if num_samples < 1:
        raise ValueError("Data must have at least one sample.")
    if num_blocks < 1:
        raise ValueError("Number of resampled blocks must be greater than or"
                         "equal to 1.")
    kf = KFold(n_splits = num_blocks)
    resampled_data = []
    for i, j in kf.split(data):
        resampled_data.append(data[i])
    return resampled_data

def jackknife(x, func, num_blocks=100):
    """ Jackknife estimate of the estimator function. """
    n = len(x)
    block_size = n // num_blocks
    idx = np.arange(0, n, block_size)
    return np.sum(func(x[idx!=i]) for i in range(n))/float(n)

def jackknife_var(x, func, num_blocks=100):
    """ Jackknife estimate of the variance of the estimator function. """
    n = len(x)
    block_size = n // num_blocks
    idx = np.arange(0, n, block_size)
    j_est = jackknife(x, func)
    return (n - 1) / (n + 0.) * np.sum(
        (func(x[idx!=i]) - j_est)**2.0 for i in range(n)
    )

def jackknife_err(y_i, y_full, num_blocks):
    if type(y_i) == list:
        y_i = np.array(y_i)
    if type(y_full) == list:
        y_full = np.array(y_full)
    try:
        err = np.sqrt((num_blocks - 1) * np.sum((y_i - y_full)**2) /
                      num_blocks)
    except ValueError:
        print("y_i.shape: {}, y_full.shape: {}".format(y_i.shape,
                                                       y_full.shape))
        raise
    return err


def jackknife_resampling(data):
    """
    Performs jackknife resampling on numpy arrays.
    Parameters
    ----------
    data : numpy.ndarray
        Original sample (1-D array) from which the jackknife resamples will be
        generated.
    Returns
    -------
    resamples : numpy.ndarray
        The i-th row is the i-th jackknife sample, i.e., the original sample
        with the i-th measurement deleted.
    """
    n = data.shape[0]
    if n <= 0:
        raise ValueError("data must contain at least one measurement.")
    resamples = np.empty([n, n-1])
    for i in range(n):
        resamples[i] = np.delete(data, i)
    return resamples

def jackknife_stats(data, statistic):
    """ Performs jackknife estimation on the basis of jackknife resamples.
    Parameters
    ----------
    data : numpy.ndarray
        Original sample (1-D array).
    statistic : function
        Any function (or vector of functions) on the basis of the measured
        data, e.g, sample mean, sample variance, etc. The jackknife estimate of
        this statistic will be returned.
    Returns
    -------
    estimate : numpy.float64 or numpy.ndarray
        The i-th element is the bias-corrected "jackknifed" estimate.
    bias : numpy.float64 or numpy.ndarray
        The i-th element is the jackknife bias.
    std_err : numpy.float64 or numpy.ndarray
        The i-th element is the jackknife standard error.
    """
    n = data.shape[0]
    if n <= 0:
        raise ValueError("data must contain at least one measurement.")
    resamples = jackknife_resampling(data)
    jack_stat = np.apply_along_axis(np.mean, 1, resamples)
    mean_jack_stat = np.mean(jack_stat, axis=0)

    std_err = np.sqrt((n - 1) * np.mean(
       (jack_stat - mean_jack_stat) * (jack_stat - mean_jack_stat), axis=0
    ))

    # bias-corrected "jackknifed estimate"
    stat_data = statistic(data)
    bias = (n - 1) * (mean_jack_stat, stat_data)
    estimate = stat_data - bias
    #     (jack_stat - mean_jack_stat) * (jack_stat - mean_jack_stat), axis=0
    #  ))
    #
    #  # bias-corrected "jackknifed estimate"
    #  stat_data = statistic(data)
    #  bias = (n - 1) * (mean_jack_stat, stat_data)
    #  estimate = stat_data - bias

    return estimate, bias, std_err

