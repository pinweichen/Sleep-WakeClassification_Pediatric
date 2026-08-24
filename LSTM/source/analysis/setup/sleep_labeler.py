import numpy as np

from source.analysis.performance.raw_performance import RawPerformance
from source.analysis.setup.sleep_label import SleepWakeLabel, ThreeClassLabel, MultiClassLabel, FourClassLabel


class SleepLabeler(object):

    @staticmethod
    def label_sleep_wake(raw_sleep_wake):
        labeled_sleep = []

        for value in raw_sleep_wake:
            p = value.copy()
            p[value > 0] = SleepWakeLabel.sleep.value
            p[value == 0] = SleepWakeLabel.wake.value
            labeled_sleep.append(p)

        return labeled_sleep

    @staticmethod
    def label_three_class(raw_sleep_wake):
        labeled_sleep = []
        for value in raw_sleep_wake:
            p = value.copy()
            p[:] = ThreeClassLabel.nrem.value
            p[value == 5] = ThreeClassLabel.rem.value
            p[value == 0] = ThreeClassLabel.wake.value
            labeled_sleep.append(p)
        return labeled_sleep

    @staticmethod
    def label_four_class(raw_sleep_wake):
        labeled_sleep = []
        for value in raw_sleep_wake:
            p = value.copy()
            p[:] = FourClassLabel.wake.value
            p[value == 1] = FourClassLabel.light.value
            p[value == 2] = FourClassLabel.light.value
            p[value == 3] = FourClassLabel.deep.value
            p[value == 5] = FourClassLabel.rem.value
            p[value == 0] = FourClassLabel.wake.value
            labeled_sleep.append(p)
        return labeled_sleep
    @staticmethod
    def label_multi_class(raw_sleep_wake):
        labeled_sleep = []
        for value in raw_sleep_wake:
            p = value.copy()
            p[:] = MultiClassLabel.r.value
            p[value == 1] = MultiClassLabel.n1.value
            p[value == 2] = MultiClassLabel.n2.value
            p[value == 3] = MultiClassLabel.n3.value
            p[value == 5] = MultiClassLabel.r.value
            p[value == 0] = MultiClassLabel.wake.value
            labeled_sleep.append(p)
        return labeled_sleep

    @staticmethod
    def label_one_vs_rest(sleep_wake_labels, positive_class):
        labeled_sleep = []

        for value in sleep_wake_labels:
            if value == positive_class:
                converted_value = 1
            else:
                converted_value = 0

            labeled_sleep.append(converted_value)

        return np.array(labeled_sleep)

    @staticmethod
    def convert_three_class_to_two(raw_performance: RawPerformance):
        raw_performance.true_labels = SleepLabeler.label_sleep_wake(raw_performance.true_labels)
        number_of_samples = np.shape(raw_performance.class_probabilities)[0]
        for index in range(number_of_samples):
            raw_performance.class_probabilities[index, 1] = raw_performance.class_probabilities[index, 1] + \
                                                            raw_performance.class_probabilities[index, 2]
        raw_performance.class_probabilities = raw_performance.class_probabilities[:, :-1]

        return raw_performance
