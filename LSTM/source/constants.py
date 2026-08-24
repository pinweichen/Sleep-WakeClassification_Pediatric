import os
from pathlib import Path

class Constants(object):

    WAKE_THRESHOLD = 0.5
    REM_THRESHOLD = 0.35

    INCLUDE_CIRCADIAN = False
    EPOCH_DURATION_IN_SECONDS = 30
    SECONDS_PER_MINUTE = 60
    SECONDS_PER_DAY = 3600 * 24
    SECONDS_PER_HOUR = 3600
    VERBOSE = True

    # AGV = actigraphy-only pipeline (no heart rate)
    DEVICE = 'agv'

    # AGV_DATA_DIR is set by run_pipeline.sh to OUTPUT_DIR so that all
    # intermediate data (labels, cropped, features, motion) lands in the
    # user-specified output directory rather than inside the repo tree.
    # Falls back to the original relative path for local / manual runs.
    _agv_data_dir = os.environ.get('AGV_DATA_DIR')
    INPUT_ROOT = (
        Path(_agv_data_dir) / 'data_processed'
        if _agv_data_dir
        else Path('../data/data_processed/')
    )

    PSG_FILE_PATH     = INPUT_ROOT / DEVICE / 'labels'
    CROPPED_FILE_PATH = INPUT_ROOT / DEVICE / 'cropped'
    FEATURE_FILE_PATH = INPUT_ROOT / DEVICE / 'features'
    MOTION_FILE_PATH  = INPUT_ROOT / DEVICE / 'motion'
    # HR_FILE_PATH intentionally removed — actigraphy-only pipeline

    if not CROPPED_FILE_PATH.exists():
        CROPPED_FILE_PATH.mkdir(parents=True, exist_ok=True)
    if not FEATURE_FILE_PATH.exists():
        FEATURE_FILE_PATH.mkdir(parents=True, exist_ok=True)

    LOWER_BOUND = -0.2
