import numpy as np
import pandas as pd
from source.constants import Constants
from preprocessing.motion_fft.motion_fft_collection import MotionFFTCollection


class MotionFFTService(object):
    @staticmethod
    def load_cropped(subject_id, symbol):
        motion_fft_path = MotionFFTService.get_cropped_vmfft_file_path(subject_id, symbol)
        counts_array = MotionFFTService.load(motion_fft_path)
        return MotionFFTCollection(subject_id=subject_id, data=counts_array, length=counts_array.shape[1])

    @staticmethod
    def load_cropped_PmaxBand(subject_id):
        motion_fft_path = MotionFFTService.get_cropped_PmaxBand_file_path(subject_id)
        counts_array = MotionFFTService.load(motion_fft_path)
        return MotionFFTCollection(subject_id=subject_id, data=counts_array)

    @staticmethod
    def load_cropped_dir_fft(subject_id, direction, symbol):
        motion_fft_path = MotionFFTService.get_cropped_dir_fft_file_path(subject_id, direction, symbol)
        counts_array = MotionFFTService.load(motion_fft_path)
        return MotionFFTCollection(subject_id=subject_id, data=counts_array, length=counts_array.shape[1])

    @staticmethod
    def load(counts_file):
        counts_array = pd.read_csv(str(counts_file), header=None).values
        return counts_array

    @staticmethod
    def get_cropped_vmfft_file_path(subject_id, symbol=''):
        return Constants.CROPPED_FILE_PATH.joinpath(subject_id + f"_motion_vmfft{symbol}.out")

    @staticmethod
    def get_cropped_PmaxBand_file_path(subject_id):
        return Constants.CROPPED_FILE_PATH.joinpath(subject_id + f"_motion_PmaxBand.out")

    @staticmethod
    def get_cropped_dir_fft_file_path(subject_id, direction, symbol=''):
        return Constants.CROPPED_FILE_PATH.joinpath(subject_id + f"_motion_{direction}fft{symbol}.out")

    @staticmethod
    def build_motion_vmfft(subject_id, data, frequencies, symbol=''):
        from scipy.signal import spectrogram
        fs = 100
        time = np.arange(np.amin(data[:, 0]), np.amax(data[:, 0]), 1.0 / fs)
        vm_data = np.interp(time, data[:, 0], data[:, 4])
        vm_df = pd.DataFrame({"time": time, "time_index": time, "vm": vm_data})

        vm_df['time_index'] = pd.to_datetime(vm_df['time_index'], unit='s', origin='unix')
        vm_df.set_index('time_index', inplace=True)
        dfs = [group for _, group in vm_df.resample('30s')]
        output = []

        for i in range(len(dfs)):
            vm_win = dfs[i].iloc[:, 1].values
            if len(vm_win) < 1500:
                continue
            # Replicate R's specgram(vm, n=Fs, Fs=Fs): NFFT=fs, noverlap=default
            f_bins, _, Sxx = spectrogram(vm_win, fs=fs, nperseg=fs,
                                         noverlap=fs // 2, nfft=fs)
            S = np.abs(Sxx)
            S_norm = S / S.max()
            f_avg = S_norm.mean(axis=1)  # average across time windows
            cur_time = dfs[i].iloc[0, 0]
            out_feature = f_avg[[fre - 1 for fre in frequencies]]
            output.append([cur_time, *out_feature])

        motion_fft_output_path = MotionFFTService.get_cropped_vmfft_file_path(subject_id, symbol)
        np.savetxt(motion_fft_output_path, output, fmt='%f', delimiter=',')

    @staticmethod
    def build_PmaxBand(subject_id, data):
        import scipy.fftpack as fftpack
        fs = 30
        time = np.arange(np.amin(data[:, 0]), np.amax(data[:, 0]), 1.0 / fs)
        vm_data = np.interp(time, data[:, 0], data[:, 4])
        vm_df = pd.DataFrame({"time": time, "vm": vm_data})
        output = []
        windows = 60
        counts = len(vm_df) // windows
        for i in range(counts):
            windows_vm = vm_data[i * windows:(i + 1) * windows]
            fourier = np.abs(fftpack.fft(windows_vm))
            frequencies = fftpack.fftfreq(windows_vm.size, d=1.0 / fs)
            f625 = [i for i in range(len(frequencies)) if frequencies[i] >= 0.6 and frequencies[i] <= 2.5]
            cut_fourier = fourier[f625]
            max_index = np.argmax(np.abs(cut_fourier))
            max_freq = frequencies[f625[max_index]]
            power_fft = cut_fourier ** 2
            max_power_fft = power_fft[max_index]
            cur_time = vm_df.iloc[i * windows:(i + 1) * windows, 0].mean()
            output.append([cur_time, max_power_fft])

        motion_fft_output_path = MotionFFTService.get_cropped_PmaxBand_file_path(subject_id)
        np.savetxt(motion_fft_output_path, output, fmt='%f', delimiter=',')

    @staticmethod
    def build_motion_direction_fft(subject_id, data, direction, frequencies, symbol=''):
        from scipy.signal import spectrogram
        dir2index = {'x': 1, 'y': 2, 'z': 3}
        ind = dir2index[direction]
        fs = 100
        time = np.arange(np.amin(data[:, 0]), np.amax(data[:, 0]), 1.0 / fs)
        dir_data = np.interp(time, data[:, 0], data[:, ind])
        x = pd.DataFrame({"time": time, "time_index": time, direction: dir_data})
        x['time_index'] = pd.to_datetime(x['time_index'], unit='s', origin='unix')
        x.set_index('time_index', inplace=True)
        dfs = [group for _, group in x.resample('30s')]
        output = []
        for i in range(len(dfs)):
            x_win = dfs[i].iloc[:, 1].values
            if len(x_win) < 1500:
                continue
            # Replicate R's specgram(x, n=Fs, Fs=Fs): NFFT=fs, noverlap=default
            f_bins, _, Sxx = spectrogram(x_win, fs=fs, nperseg=fs,
                                          noverlap=fs // 2, nfft=fs)
            S = np.abs(Sxx)
            S_norm = S / S.max()
            f_avg = S_norm.mean(axis=1)  # average across time windows
            cur_time = dfs[i].iloc[0, 0]
            out_feature = f_avg[[fre - 1 for fre in frequencies]]
            output.append([cur_time, *out_feature])

        motion_fft_output_path = MotionFFTService.get_cropped_dir_fft_file_path(subject_id, direction, symbol)
        np.savetxt(motion_fft_output_path, output, fmt='%f', delimiter=',')

    @staticmethod
    def max2epochs(data, fs, epoch):
        data = data.flatten()

        seconds = int(np.floor(np.shape(data)[0] / fs))
        data = np.abs(data)
        data = data[0:int(seconds * fs)]

        data = data.reshape(fs, seconds, order='F').copy()

        data = data.max(0)
        data = data.flatten()
        N = np.shape(data)[0]
        num_epochs = int(np.floor(N / epoch))
        data = data[0:(num_epochs * epoch)]

        data = data.reshape(epoch, num_epochs, order='F').copy()
        epoch_data = np.sum(data, axis=0)
        epoch_data = epoch_data.flatten()

        return epoch_data

    @staticmethod
    def crop(motion_fft_collection, interval):
        subject_id = motion_fft_collection.subject_id
        timestamps = motion_fft_collection.timestamps
        valid_indices = ((timestamps >= interval.start_time)
                         & (timestamps < interval.end_time)).nonzero()[0]

        cropped_data = motion_fft_collection.data[valid_indices, :]
        return MotionFFTCollection(subject_id=subject_id, data=cropped_data)
