import csv

import numpy as np
import pandas as pd

from source import utils
from source.constants import Constants
from preprocessing.epoch import Epoch
from preprocessing.psg.psg_converter import PSGConverter
from preprocessing.psg.psg_raw_data_collection import PSGRawDataCollection
from preprocessing.psg.stage_item import StageItem


class PSGService(object):

    @staticmethod
    def read_precleaned(subject_id):
        psg_path = Constants.PSG_FILE_PATH.joinpath(subject_id + "_labeled.csv")
        data = []
        with open(psg_path, 'rt') as csv_file:
            file_reader = csv.reader(csv_file, delimiter=',', quotechar='|')
            next(file_reader)  # skip header
            rows = list(file_reader)

        if len(rows) < 2:
            return PSGRawDataCollection(subject_id=subject_id, data=data)

        # Auto-detect how many sensor rows make up one 30-second PSG epoch.
        # Label files from high-frequency devices (e.g. 50 Hz actigraphy) have
        # one row per sensor sample, not one row per epoch.  The Time column
        # (last column) stores elapsed seconds; two consecutive rows reveal the
        # sampling interval.  For a 50 Hz file: 0.02 s/sample → 1500 rows/epoch.
        # For an already-epoch-level file: ~30 s/row → rows_per_epoch = 1.
        time_increment = float(rows[1][-1]) - float(rows[0][-1])
        if 0 < time_increment < 30:
            rows_per_epoch = max(1, round(30.0 / time_increment))
        else:
            rows_per_epoch = 1

        # Stride through the rows, taking the first sample of each 30-s epoch.
        epoch_rows = rows[::rows_per_epoch]
        start_time = float(epoch_rows[0][-1])

        for count, row in enumerate(epoch_rows):
            score = int(row[2])
            timestamp = start_time + count * 30
            epoch = Epoch(timestamp=timestamp, index=(1 + count))
            data.append(StageItem(epoch=epoch, stage=PSGConverter.get_label_from_int(score)))

        return PSGRawDataCollection(subject_id=subject_id, data=data)

    @staticmethod
    def crop(psg_raw_collection, interval):
        subject_id = psg_raw_collection.subject_id

        stage_items = []
        for stage_item in psg_raw_collection.data:
            timestamp = stage_item.epoch.timestamp
            if interval.start_time <= timestamp < interval.end_time:
                stage_items.append(stage_item)

        return PSGRawDataCollection(subject_id=subject_id, data=stage_items)

    @staticmethod
    def write(psg_raw_data_collection):
        data_array = []

        for index in range(len(psg_raw_data_collection.data)):
            stage_item = psg_raw_data_collection.data[index]
            data_array.append([stage_item.epoch.timestamp, stage_item.stage.value])

        np_psg_array = np.array(data_array)
        psg_output_path = Constants.CROPPED_FILE_PATH.joinpath(psg_raw_data_collection.subject_id + "_cleaned_psg.out")
        if not Constants.CROPPED_FILE_PATH.exists():
            Constants.CROPPED_FILE_PATH.mkdir(parents=True)
        np.savetxt(psg_output_path, np_psg_array, fmt='%f',delimiter=',')

    @staticmethod
    def load_cropped_array(subject_id):
        cropped_psg_path = Constants.CROPPED_FILE_PATH.joinpath(subject_id + "_cleaned_psg.out")
        return pd.read_csv(str(cropped_psg_path), delimiter=',',header=None).values

    @staticmethod
    def load_cropped(subject_id):
        cropped_array = PSGService.load_cropped_array(subject_id)
        stage_items = []

        for row in range(np.shape(cropped_array)[0]):
            value = cropped_array[row, 1]
            stage_items.append(StageItem(epoch=Epoch(timestamp=cropped_array[row, 0], index=row),
                                         stage=PSGConverter.get_label_from_int(value)))

        return PSGRawDataCollection(subject_id=subject_id, data=stage_items)
