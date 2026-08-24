from pathlib import Path
from source.constants import Constants
from preprocessing.motion_fft.motion_fft_feature_service import MotionFFTFeatureService
from preprocessing.psg.psg_label_service import PSGLabelService
from preprocessing.raw_data_processor import RawDataProcessor
# HeartRateFeatureService removed — actigraphy-only (AGV) pipeline

feature_name = {
    'biobank': ['enmoTrunc', 'enmoAbs', 'xMean', 'yMean', 'zMean', 'xRange', 'yRange', 'zRange', 'xStd',
                'yStd', 'zStd', 'xyCov', 'xzCov', 'yzCov', 'entropy', 'MPD', 'skew', 'kurt', 'avgArmAngel',
                'avgArmAngelAbsDiff', 'f1', 'p1', 'f2', 'p2', 'f625', 'p625', 'totalPower'],
    'tlbc': ["fMean", "fStd", "fCoefVariation", "fMedian", "fMin", "fMax", "f25thP", "f75thP", "fAutocorr",
             "fCorrxy", "fCorrxz", "fCorryz", "fAvgRoll", "fAvgPitch", "fAvgYaw", "fSdRoll", "fSdPitch", "fSdYaw",
             "fRollG", "fPitchG", "fYawG", "fFmax", "fPmax", "fFmaxBand", "fPmaxBand", "fEntropy", "vMFFT0", "FFT1",
             "vFFT2", "vFFT3", "vFFT4", "vFFT5", "vFFT6", "vFFT7", "vFFT8", "vFFT9", "vFFT10", "vFFT11", "vFFT12",
             "vFFT13", "vFFT14"],
    'ggir': ['BFEN', 'LFEN', 'LFENMO', 'HFEN', 'HFENplus', 'roll_med_acc_x', 'roll_med_acc_y', 'roll_med_acc_z',
             'dev_roll_med_acc_x', 'dev_roll_med_acc_y', 'dev_roll_med_acc_z', 'angle_x', 'angle_y', 'angle_z',
             'ENMO', 'MAD', 'EN', 'ENMOa']
}


class FeatureBuilder(object):

    @staticmethod
    def _feature_done_marker(subject_id) -> Path:
        """Zero-byte file written after all feature files for a subject complete."""
        return Constants.FEATURE_FILE_PATH / f"{subject_id}_feature_done"

    @staticmethod
    def build(subject_id):
        if Constants.VERBOSE:
            print(f"{subject_id} Getting valid epochs...")

        # Skip only when the completion marker exists.  The marker is written
        # after every feature file is fully saved, so a job killed mid-write
        # will leave partial .out files but no marker, forcing a clean re-run.
        if FeatureBuilder._feature_done_marker(subject_id).exists():
            print(f"{subject_id} skip (feature completion marker found)")
            return

        valid_epochs = RawDataProcessor.get_valid_epochs(subject_id)
        if len(valid_epochs) < 200:
            print(f'remove {subject_id} valid_epochs: {len(valid_epochs)}')
            return None
        if Constants.VERBOSE:
            print(f"{subject_id} Building features...")
        FeatureBuilder.build_labels(subject_id, valid_epochs)
        FeatureBuilder.build_from_wearables(subject_id, valid_epochs)

        # All feature files written — touch the completion marker.
        FeatureBuilder._feature_done_marker(subject_id).touch()

    @staticmethod
    def build_labels(subject_id, valid_epochs):
        psg_labels = PSGLabelService.build(subject_id, valid_epochs)
        psg_timestamps = PSGLabelService.build_timestamps(valid_epochs)
        PSGLabelService.write(subject_id, psg_labels)
        PSGLabelService.write_timestamps(subject_id, psg_timestamps)

    @staticmethod
    def build_from_wearables(subject_id, valid_epochs):
        # Heart rate feature removed — AGV pipeline uses motion only
        motion_xfft30_feature = MotionFFTFeatureService.build_direction_fft(subject_id, valid_epochs, 'x', '1-30')
        motion_yfft30_feature = MotionFFTFeatureService.build_direction_fft(subject_id, valid_epochs, 'y', '1-30')
        motion_zfft30_feature = MotionFFTFeatureService.build_direction_fft(subject_id, valid_epochs, 'z', '1-30')
        motion_vmfft30_feature = MotionFFTFeatureService.build_vmfft(subject_id, valid_epochs, '1-30')

        MotionFFTFeatureService.write_dir_fft(subject_id, motion_xfft30_feature, 'x', '1-30')
        MotionFFTFeatureService.write_dir_fft(subject_id, motion_yfft30_feature, 'y', '1-30')
        MotionFFTFeatureService.write_dir_fft(subject_id, motion_zfft30_feature, 'z', '1-30')
        MotionFFTFeatureService.write_vmfft(subject_id, motion_vmfft30_feature, '1-30')
