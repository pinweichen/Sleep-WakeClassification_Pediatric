import numpy as np
import pandas as pd

from source.constants import Constants
from preprocessing.psg.psg_service import PSGService


class PSGLabelService(object):
    @staticmethod
    def load(subject_id):
        psg_label_path = PSGLabelService.get_path(subject_id)
        feature = pd.read_csv(str(psg_label_path), header=None).values
        return feature

    @staticmethod
    def get_path(subject_id):
        return Constants.FEATURE_FILE_PATH.joinpath(subject_id + '_psg_labels.out')

    @staticmethod
    def get_timestamps_path(subject_id):
        """Path to the per-epoch Unix timestamp file (one value per 30-s epoch)."""
        return Constants.FEATURE_FILE_PATH.joinpath(subject_id + '_psg_timestamps.out')

    @staticmethod
    def load_timestamps(subject_id):
        """Load the saved epoch-level Unix timestamps for a subject."""
        path = PSGLabelService.get_timestamps_path(subject_id)
        return pd.read_csv(str(path), header=None).values.flatten()

    @staticmethod
    def build(subject_id, valid_epochs):
        psg_array = PSGService.load_cropped_array(subject_id)
        labels = []
        for epoch in valid_epochs:
            value = np.interp(epoch.timestamp, psg_array[:, 0], psg_array[:, 1])
            labels.append(value)
        return np.array(labels)

    @staticmethod
    def build_timestamps(valid_epochs):
        """Extract the Unix timestamp from each valid 30-second epoch."""
        return np.array([epoch.timestamp for epoch in valid_epochs])

    @staticmethod
    def write(subject_id, labels):
        psg_labels_path = PSGLabelService.get_path(subject_id)
        np.savetxt(psg_labels_path, labels, fmt='%f')

    @staticmethod
    def write_timestamps(subject_id, timestamps):
        """Persist per-epoch timestamps alongside the label file."""
        path = PSGLabelService.get_timestamps_path(subject_id)
        np.savetxt(path, timestamps, fmt='%f')
