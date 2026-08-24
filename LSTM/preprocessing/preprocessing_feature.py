import os
import time
import sys
from pathlib import Path
import pandas as pd
sys.path.append('..')
from source.analysis.setup.subject_builder import SubjectBuilder
from source.constants import Constants
from preprocessing.feature_builder import FeatureBuilder
from preprocessing.raw_data_processor import RawDataProcessor
from joblib import Parallel, delayed


def run_preprocessing(subject_set):
    start_time = time.time()
    parallel = Parallel(n_jobs=5)
    parallel(delayed(RawDataProcessor.crop_all)(str(subject_set[i])) for i in (range(len(subject_set))))

    parallel(delayed(FeatureBuilder.build)(str(subject_set[i])) for i in (range(len(subject_set))))

    end_time = time.time()
    print("Execution took " + str((end_time - start_time) / 60) + " minutes")


if __name__ == '__main__':
    # AGV_DATA_DIR is set by run_pipeline.sh to OUTPUT_DIR.
    # agv_ids.csv is written there by data_ingestion.py (Step 1).
    # Falls back to the original repo-relative path for local/manual runs.
    _agv_data_dir = os.environ.get('AGV_DATA_DIR')
    if _agv_data_dir:
        ids_csv = Path(_agv_data_dir) / f'{Constants.DEVICE}_ids.csv'
    else:
        ids_csv = Path(f'../data/{Constants.DEVICE}_ids.csv')

    subject_ids = pd.read_csv(ids_csv)['subject'].tolist()
    run_preprocessing(subject_ids)
