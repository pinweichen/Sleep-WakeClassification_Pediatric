import numpy as np
import pandas as pd

from source import utils
from source.constants import Constants
from preprocessing.motion_fft.motion_fft_service import MotionFFTService
from preprocessing.epoch import Epoch


class MotionFFTFeatureService(object):
    WINDOW_SIZE = 15

    @staticmethod
    def load_vmfft(subject_id, symbol):
        motion_fft_feature_path = MotionFFTFeatureService.get_path(subject_id, symbol)
        feature = pd.read_csv(str(motion_fft_feature_path), header=None, sep=' ').values
        return feature

    @staticmethod
    def load_PmaxBand(subject_id):
        motion_fft_feature_path = MotionFFTFeatureService.get_path_PmaxBand(subject_id)
        feature = pd.read_csv(str(motion_fft_feature_path)).values
        return feature

    @staticmethod
    def load_dir_fft(subject_id, direction, symbol):
        motion_fft_feature_path = MotionFFTFeatureService.get_path_dir_fft(subject_id, direction, symbol)
        feature = pd.read_csv(str(motion_fft_feature_path), header=None, sep=' ').values
        return feature

    @staticmethod
    def get_path(subject_id, symbol):
        return Constants.FEATURE_FILE_PATH.joinpath(subject_id + f'_motion_vmfft{symbol}_feature.out')

    @staticmethod
    def get_path_PmaxBand(subject_id):
        return Constants.FEATURE_FILE_PATH.joinpath(subject_id + f'_motion_PmaxBand_feature.out')

    @staticmethod
    def get_path_dir_fft(subject_id, direction, symbol):
        return Constants.FEATURE_FILE_PATH.joinpath(subject_id + f'_motion_{direction}fft{symbol}_feature.out')

    @staticmethod
    def write_vmfft(subject_id, feature, symbol):
        motion_fft_feature_path = MotionFFTFeatureService.get_path(subject_id, symbol)
        np.savetxt(motion_fft_feature_path, feature.reshape(feature.shape[0], -1), fmt='%f')

    @staticmethod
    def write_PmaxBand(subject_id, feature):
        motion_fft_feature_path = MotionFFTFeatureService.get_path_PmaxBand(subject_id)
        np.savetxt(motion_fft_feature_path, feature, fmt='%f')

    @staticmethod
    def write_dir_fft(subject_id, feature, direction, frequency):
        motion_fft_feature_path = MotionFFTFeatureService.get_path_dir_fft(subject_id, direction, frequency)
        np.savetxt(motion_fft_feature_path, feature.reshape(feature.shape[0], -1), fmt='%f')

    @staticmethod
    def get_window(timestamps, epoch):
        start_time = epoch.timestamp - Epoch.DURATION//2
        end_time = epoch.timestamp + Epoch.DURATION//2
        timestamps_ravel = timestamps.ravel()
        indices_in_range = np.unravel_index(np.where((timestamps_ravel > start_time) & (timestamps_ravel < end_time)),
                                            timestamps.shape)
        return indices_in_range[0][0]

    @staticmethod
    def build_vmfft(subject_id, valid_epochs, symbol):
        motion_fft_collection = MotionFFTService.load_cropped(subject_id, symbol)
        return MotionFFTFeatureService.build_from_collection(motion_fft_collection, valid_epochs)

    @staticmethod
    def build_PmaxBand(subject_id, valid_epochs):
        motion_fft_collection = MotionFFTService.load_cropped_PmaxBand(subject_id)
        return MotionFFTFeatureService.build_from_collection(motion_fft_collection, valid_epochs)

    @staticmethod
    def build_direction_fft(subject_id, valid_epochs, direction, symbol):
        motion_fft_collection = MotionFFTService.load_cropped_dir_fft(subject_id, direction, symbol)
        return MotionFFTFeatureService.build_from_collection(motion_fft_collection, valid_epochs)

    @staticmethod
    def build_from_collection(motion_fft_collection, valid_epochs):
        fft_features = []

        interpolated_timestamps, interpolated_features_list = MotionFFTFeatureService.interpolate(
            motion_fft_collection)

        for epoch in valid_epochs:
            indices_in_range = MotionFFTFeatureService.get_window(interpolated_timestamps, epoch)
            feature = []
            for interpolated_features in interpolated_features_list:
                motion_fft_in_range = interpolated_features[indices_in_range]
                motion_feature = MotionFFTFeatureService.get_feature(motion_fft_in_range)
                feature.append(motion_feature)
            fft_features.append(feature)

        return np.array(fft_features)

    @staticmethod
    def get_feature(count_values):
        convolution = utils.smooth_gauss(count_values.flatten(), np.shape(count_values.flatten())[0])
        return np.array([convolution])

    @staticmethod
    def interpolate(motion_fft_collection):
        timestamps = motion_fft_collection.timestamps.flatten()
        dim = motion_fft_collection.values.shape[1]

        interpolated_timestamps = np.arange(np.amin(timestamps),
                                            np.amax(timestamps), 1)
        interpolated_counts_list = []
        for i in range(dim):
            ggir_values = motion_fft_collection.values[:, i].flatten()
            interpolated_counts_list.append(np.interp(interpolated_timestamps, timestamps, ggir_values))
        return interpolated_timestamps, interpolated_counts_list
