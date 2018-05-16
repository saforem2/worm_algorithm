import os
import numpy as np
from scipy.misc import toimage

def unpickle(file):
    """Method for opening CIFAR-10 dataset. Returns dictionary."""
    import pickle
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict

def reshape_cifar10_image(image):
    """Method for reshaping CIFAR-10 images into array of shape:
    [num_images, 32, 32, 3]. """
    new_image = np.zeros((32, 32, 3))
    new_image[:, :, 0] = image[:1024].reshape(32, 32)
    new_image[:, :, 1] = image[1024:2048].reshape(32, 32)
    new_image[:, :, 2] = image[2048:3072].reshape(32, 32)
    return new_image

def  process_cifar10_image(image):
    """Reshape and convert CIFAR-10 images to greyscale."""
    reshaped_image = reshape_cifar10_image(image)
    image_ = toimage(reshaped_image)
    image_ = image_.convert("L")
    image_arr = np.array(image_.getdata()).reshape(image_.size[1],
                                                   image_.size[0])
    return image_arr

def get_plot_num(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    existing = [int(i.split('_')[-1].rstrip('.png')) for i in
                os.listdir(out_dir) if i.endswith('.png')]
    try:
        latest_num = max(existing)
    except ValueError:
        latest_num = 0
    return latest_num + 1

def block_resampling(data, num_blocks):
    """Block-resample data to return num_blocks samples of original data. """
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    num_samples = data.shape[0]
    if num_samples < 1:
        raise ValueError("Data must have at least one sample.")
    if num_blocks < 1:
        raise ValueError("Number of resampled blocks must be greater than or"
                         "equal to 1")
    kf = KFold(n_splits == num_blocks)
    resampled_data = []
    for i, j in kf.split(data):
        resampled_data.append(data[i])
    return resampled_data

def jackknife(x, func, num_blocks=100):
    """Jackknife estimate of the estimator functiion."""
    n = len(x)
    block_size = n // num_blocks
    idx = np.arange(0, n, block_size)
    return np.sum(func(x[idx!=i]) for i in range(n)) / float(n)

def jackknife_var(x, func, num_blocks=100):
    """Jackknife estimiate of the variance of the estimator function."""
    n = len(x)
    block_size = n // num_blocks
    idx = np.arange(0, n, block_size)
    j_est = jackknife(x, func)
    return (n - 1) / (n + 0.) * np.sum((func(x[idx!=i]) - j_est)**2.0 for i in
                                      range(n))

def jackknife_err(y_i, y_full, num_blocks):
    if isinstance(y_i, list):
        y_i = np.array(y_i)
    if isinstance(y_full, list):
        y_full = np.array(y_full)
    try:
        err = np.sqrt((num_blocks - 1) * np.sum((y_i - y_full)**2) / num_blocks)
    except ValueError:
        raise ValueError(f"y_i.shape: {y_i.shape}, y_full.shape:"
                         f"{y_full.shape}")
    return err

def jackknife_resampling(data):
    """Performs jackknife resampling on numpy arrays."""

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
