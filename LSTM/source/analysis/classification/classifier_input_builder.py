import numpy as np
from sklearn.impute import KNNImputer

from source.analysis.setup.sleep_labeler import SleepLabeler


class ClassifierInputBuilder(object):

    @staticmethod
    def get_array(subject_ids, subject_dictionary, feature_set):

        all_subjects_features = []
        all_subjects_labels = []

        for subject_id in subject_ids:
            subject_features = []
            subject = subject_dictionary[subject_id]
            feature_dictionary = subject.feature_dictionary

            for feature in feature_set:
                feature_data = feature_dictionary[feature]
                feature_data = np.expand_dims(feature_data, axis=-1)
                subject_features.append(feature_data)

            subject_features = np.concatenate(subject_features, axis=-1)
            subject_labels = subject.labeled_sleep.reshape(-1)

            all_subjects_features.append(subject_features)
            all_subjects_labels.append(subject_labels)

        return all_subjects_features, all_subjects_labels

    @staticmethod
    def impute(all_subjects_features):
        """
        Apply KNN imputation for NaN values only (no normalization).

        FFT spectral power values are used as-is — the normalized power
        density from the spectrogram is already meaningful and should not
        be rescaled with Z-score normalization.

        Returns
        -------
        all_subjects_features : list of np.ndarray
            Imputed feature arrays (same shapes as input).
        imputer : KNNImputer
            Fitted imputer (needed to transform test data consistently).
        """
        shapes = [sf.shape for sf in all_subjects_features]
        n_channels = shapes[0][1] if len(shapes[0]) == 3 else 1
        n_features = shapes[0][-1]

        flat_list = []
        for sf in all_subjects_features:
            flat_list.append(sf.reshape(-1, n_features))
        flat_all = np.concatenate(flat_list, axis=0)

        # KNN imputation only
        imputer = KNNImputer(n_neighbors=5)
        flat_all = imputer.fit_transform(flat_all)

        # Reshape back to per-subject arrays
        result = []
        offset = 0
        for i, sf in enumerate(all_subjects_features):
            orig_shape = shapes[i]
            n_rows = orig_shape[0] * (n_channels if len(orig_shape) == 3 else 1)
            chunk = flat_all[offset:offset + n_rows]
            result.append(chunk.reshape(orig_shape))
            offset += n_rows

        return result, imputer

    @staticmethod
    def transform_with_fitted_imputer(all_subjects_features, imputer):
        """
        Apply a previously fitted KNN imputer to new data (e.g., test subjects).
        No normalization is applied.
        """
        shapes = [sf.shape for sf in all_subjects_features]
        n_channels = shapes[0][1] if len(shapes[0]) == 3 else 1
        n_features = shapes[0][-1]

        flat_list = []
        for sf in all_subjects_features:
            flat_list.append(sf.reshape(-1, n_features))
        flat_all = np.concatenate(flat_list, axis=0)

        flat_all = imputer.transform(flat_all)

        result = []
        offset = 0
        for i, sf in enumerate(all_subjects_features):
            orig_shape = shapes[i]
            n_rows = orig_shape[0] * (n_channels if len(orig_shape) == 3 else 1)
            chunk = flat_all[offset:offset + n_rows]
            result.append(chunk.reshape(orig_shape))
            offset += n_rows

        return result

    @staticmethod
    def get_sleep_wake_inputs(subject_ids, subject_dictionary, feature_set):
        values, raw_labels = ClassifierInputBuilder.get_array(subject_ids, subject_dictionary, feature_set)
        processed_labels = SleepLabeler.label_sleep_wake(raw_labels)
        return values, processed_labels


    @staticmethod
    def get_four_class_inputs(subject_ids, subject_dictionary, feature_set):
        values, raw_labels = ClassifierInputBuilder.get_array(subject_ids, subject_dictionary, feature_set)
        processed_labels = SleepLabeler.label_four_class(raw_labels)
        return values, processed_labels


    @staticmethod
    def __append_feature(array, feature):
        if len(np.shape(feature)) < 2:
            feature = np.transpose([feature])
        if np.shape(array)[0] == 0:
            array = feature
        else:
            array = np.hstack((array, feature))

        return array

    @staticmethod
    def __stack(combined_array, new_array):
        if np.shape(combined_array)[0] == 0:
            combined_array = new_array
        else:
            combined_array = np.vstack((combined_array, new_array))
        return combined_array
