import random
import math
import numpy as np

from source.analysis.setup.feature_type import FeatureType
from source.analysis.setup.data_split import DataSplit


class TrainTestSplitter(object):

    @staticmethod
    def leave_one_out(subject_ids):
        splits = []

        for index in range(len(subject_ids)):
            training_set = subject_ids.copy()
            testing_set = [training_set.pop(index)]

            splits.append(DataSplit(training_set=training_set, testing_set=testing_set))

        return splits

    @staticmethod
    def by_fraction(subject_ids, test_fraction, number_of_splits):

        test_index = int(np.round(test_fraction * len(subject_ids)))

        splits = []
        for trial in range(number_of_splits):
            random.shuffle(subject_ids)

            training_set = subject_ids.copy()
            testing_set = []
            for index in range(test_index):
                testing_set.append(training_set.pop(0))

            splits.append(DataSplit(training_set=training_set, testing_set=testing_set))

        return splits

    @staticmethod
    def by_number(subject_ids, test_index, seed=1):
        number_of_splits = math.ceil(len(subject_ids) / test_index)
        splits = []
        random.seed(seed)
        random.shuffle(subject_ids)
        for trial in range(number_of_splits):
            copy_set = subject_ids.copy()
            testing_set = copy_set[trial * test_index:(trial + 1) * test_index]
            training_set = [i for i in copy_set if i not in testing_set]
            splits.append(DataSplit(training_set=training_set, testing_set=testing_set))
        return splits

    @staticmethod
    def by_number_age(subject_dictionary, number_of_splits=10):
        splits = []
        sub_age = {}
        for k, v in subject_dictionary.items():
            sub_age[k] = v.feature_dictionary[FeatureType.age].mean()

        sorted_dict = dict(sorted(sub_age.items(), key=lambda item: item[1]))
        sub_age = np.array([[k, sub_age[k]] for k in sorted_dict])
        splits_index = np.array([i % number_of_splits for i in range(len(subject_dictionary))])

        for i in range(number_of_splits):
            testing_set = sub_age[splits_index == i, 0]
            training_set = sub_age[splits_index != i, 0]
            splits.append(DataSplit(training_set=training_set, testing_set=testing_set))
        return splits

    @staticmethod
    def by_class_number(subject_dictionary, number_of_splits=10):
        splits = []
        sub_age = {}
        for k, v in subject_dictionary.items():
            sub_age[k] = v.feature_dictionary[FeatureType.age].mean()

        sorted_dict = dict(sorted(sub_age.items(), key=lambda item: item[1]))
        sub_age = np.array([[k, sub_age[k]] for k in sorted_dict])
        splits_index = np.array([i % number_of_splits for i in range(len(subject_dictionary))])

        for i in range(number_of_splits):
            testing_set = sub_age[splits_index == i, 0]
            training_set = sub_age[splits_index != i, 0]
            splits.append(DataSplit(training_set=training_set, testing_set=testing_set))
        return splits
