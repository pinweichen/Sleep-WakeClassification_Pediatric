from pathlib import Path

import numpy as np
from source.analysis.setup.feature_type import FeatureType


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_lstm_feature_fft30_sets():
    # Heart rate removed — actigraphy-only (AGV) pipeline.
    # feature_dim = len(feature_sets[0]) = 4  (x, y, z, vm)
    # Model input shape per epoch: (30_fft_bins, 4_feature_types)
    return [
        [FeatureType.motion_xfft1_30, FeatureType.motion_yfft1_30, FeatureType.motion_zfft1_30,
         FeatureType.motion_vmfft1_30]]


def smooth_gauss(y, box_pts):
    box = np.ones(box_pts) / box_pts
    mu = int(box_pts / 2.0)
    sigma = 50  # seconds

    for ind in range(0, box_pts):
        box[ind] = np.exp(-1 / 2 * (((ind - mu) / sigma) ** 2))

    box = box / np.sum(box)
    sum_value = 0
    for ind in range(0, box_pts):
        sum_value += box[ind] * y[ind]

    return sum_value


def convolve_with_dog(y, box_pts):
    y = y - np.mean(y)
    box = np.ones(box_pts) / box_pts

    mu1 = int(box_pts / 2.0)
    sigma1 = 120

    mu2 = int(box_pts / 2.0)
    sigma2 = 600

    scalar = 0.75

    for ind in range(0, box_pts):
        box[ind] = np.exp(-1 / 2 * (((ind - mu1) / sigma1) ** 2)) - scalar * np.exp(
            -1 / 2 * (((ind - mu2) / sigma2) ** 2))

    y = np.insert(y, 0, np.flip(y[0:int(box_pts / 2)]))  # Pad by repeating boundary conditions
    y = np.insert(y, len(y) - 1, np.flip(y[int(-box_pts / 2):]))
    y_smooth = np.convolve(y, box, mode='valid')

    return y_smooth


def remove_repeats(array):
    array_no_repeats = np.unique(array, axis=0)
    array_no_repeats = array_no_repeats[np.argsort(array_no_repeats[:, 0])]
    return array_no_repeats


def remove_nans(array):
    array = array[~np.isnan(array).any(axis=1)]
    array = array[~np.isinf(array).any(axis=1)]
    return array
