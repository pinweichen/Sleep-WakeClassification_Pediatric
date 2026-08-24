import numpy as np
from source import utils
from source.constants import Constants
from preprocessing.epoch import Epoch
from preprocessing.interval import Interval
from preprocessing.motion.motion_service import MotionService
from preprocessing.psg.psg_service import PSGService
from preprocessing.motion_fft.motion_fft_service import MotionFFTService
from source.sleep_stage import SleepStage
# HeartRateService removed — actigraphy-only (AGV) pipeline


class RawDataProcessor:
    BASE_FILE_PATH = utils.get_project_root().joinpath('outputs/cropped/')

    @staticmethod
    def _crop_done_marker(subject_id):
        """Path to the zero-byte file written after crop_all completes fully."""
        return Constants.CROPPED_FILE_PATH / f"{subject_id}_crop_done"

    @staticmethod
    def crop_all(subject_id):
        # Skip only if the completion marker exists.  The marker is written as
        # the very last act of crop_all, so it is absent if the job was killed
        # mid-run (even if partial .out files exist).
        if RawDataProcessor._crop_done_marker(subject_id).exists():
            print(f"{subject_id} skip crop_all (completion marker found)")
            return

        print("Cropping data from subject " + subject_id + "...")

        psg_raw_collection = PSGService.read_precleaned(subject_id)
        motion_collection = MotionService.load_raw(subject_id)

        # Intersect over PSG and motion only (no heart rate)
        valid_interval = RawDataProcessor.get_intersecting_interval([psg_raw_collection,
                                                                     motion_collection])

        psg_raw_collection = PSGService.crop(psg_raw_collection, valid_interval)
        motion_collection = MotionService.crop(motion_collection, valid_interval)

        MotionFFTService.build_motion_direction_fft(subject_id, motion_collection.data, 'x', [i + 1 for i in range(30)],
                                                    '1-30')
        MotionFFTService.build_motion_direction_fft(subject_id, motion_collection.data, 'y', [i + 1 for i in range(30)],
                                                    '1-30')
        MotionFFTService.build_motion_direction_fft(subject_id, motion_collection.data, 'z', [i + 1 for i in range(30)],
                                                    '1-30')
        MotionFFTService.build_motion_vmfft(subject_id, motion_collection.data, [i + 1 for i in range(30)], '1-30')

        PSGService.write(psg_raw_collection)
        MotionService.write(motion_collection)
        # HeartRateService.write removed — no HR in AGV pipeline

        # All writes succeeded — touch the completion marker.
        # crop_all will be skipped on any future re-run for this subject.
        RawDataProcessor._crop_done_marker(subject_id).touch()

    @staticmethod
    def get_intersecting_interval(collection_list):
        start_times = []
        end_times = []
        for collection in collection_list:
            interval = collection.get_interval()
            start_times.append(interval.start_time)
            end_times.append(interval.end_time)

        return Interval(start_time=max(start_times), end_time=min(end_times))

    @staticmethod
    def get_valid_epochs(subject_id):
        psg_collection = PSGService.load_cropped(subject_id)
        motion_collection = MotionService.load_cropped(subject_id)
        # heart_rate_collection removed — AGV pipeline validates on motion only

        start_time = psg_collection.data[0].epoch.timestamp
        motion_epoch_dictionary = RawDataProcessor.get_valid_epoch_dictionary(motion_collection.timestamps,
                                                                              start_time)
        valid_epochs = []
        for stage_item in psg_collection.data:
            epoch = stage_item.epoch
            # Only require motion coverage (no HR check)
            if epoch.timestamp in motion_epoch_dictionary                     and stage_item.stage != SleepStage.unscored:
                valid_epochs.append(epoch)

        return valid_epochs

    @staticmethod
    def get_valid_epoch_dictionary(timestamps, start_time):
        epoch_dictionary = {}

        for ind in range(np.shape(timestamps)[0]):
            time = timestamps[ind]
            floored_timestamp = time - np.mod(time - start_time, Epoch.DURATION)

            epoch_dictionary[floored_timestamp] = True

        return epoch_dictionary
