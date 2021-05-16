from abc import ABC

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import scipy.io as sio
import numpy as np
import os


class CustomDataSet(Dataset):
    def __init__(self, x, y):
        self.X = x
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]


class LoadDataSet:
    def __init__(self, dir_name, path_mode="min_delay", rnd_state=3, version=2.1):
        # load the data
        self.dir_name = dir_name
        self.dir_path = os.path.abspath("../../"+self.dir_name+"pathwisedata")
        self.path_mode = path_mode
        self.rnd_state = rnd_state
        self.version = version
        np.random.seed(self.rnd_state)
        # feature data
        rx_power_tensor = sio.loadmat(os.path.join(self.dir_path, "all_rxpower_tensor_paths"))['rx_power_tensor']
        toa_tensor = sio.loadmat(os.path.join(self.dir_path, "all_toa_tensor_paths"))['toa_tensor']
        zenith_aoa_tensor = sio.loadmat(os.path.join(self.dir_path, "all_zenith_aoa_tensor_paths"))['zenith_aoa_tensor']
        azimuth_aoa_tensor = sio.loadmat(os.path.join(self.dir_path, "all_azimuth_aoa_tensor_paths"))['azimuth_aoa_tensor']
        zenith_aod_tensor = sio.loadmat(os.path.join(self.dir_path, "all_zenith_aod_tensor_paths"))['zenith_aod_tensor']
        azimuth_aod_tensor = sio.loadmat(os.path.join(self.dir_path, "all_azimuth_aod_tensor_paths"))['azimuth_aod_tensor']

        self._compute_indices(toa_tensor)

        self.dir_path = os.path.abspath("../../"+self.dir_name)
        true_cord_tensor = sio.loadmat(os.path.join(self.dir_path, "all_true_tensor"))['true_cord_tensor']
        bs_coords = sio.loadmat(os.path.join(self.dir_path, "BScords"))
        min_rx_cords = np.array([373.5530, 295.8490, 68.1720])
        bs1_coord = bs_coords['bs1cords'] - min_rx_cords
        bs2_coord = bs_coords['bs2cords'] - min_rx_cords
        bs3_coord = bs_coords['bs3cords'] - min_rx_cords
        bs4_coord = bs_coords['bs4cords'] - min_rx_cords
        self.bss_coords = np.vstack((bs1_coord, bs2_coord, bs3_coord, bs4_coord))

        # reshape the data over the number of trajectories
        self.tot_points = rx_power_tensor.shape[0] * rx_power_tensor.shape[1]

        # selecting a sub-set of the data for each path
        toa_tensor = self._get_path_data(toa_tensor*1e3)  # convert the toa to milliseconds
        rx_power_tensor = self._get_path_data(rx_power_tensor)  # convert power to milliwatt
        zenith_aoa_tensor = self._get_path_data(zenith_aoa_tensor)
        azimuth_aoa_tensor = self._get_path_data(azimuth_aoa_tensor)
        zenith_aod_tensor = self._get_path_data(zenith_aod_tensor)
        azimuth_aod_tensor = self._get_path_data(azimuth_aod_tensor)

        # compute the SNR for the addition of noise to the data
        bandwidth = 400e6
        k_boltz = 1.38e-23
        noise_figure = 10 ** 0.9
        temp = 298
        noise_power = k_boltz * bandwidth * noise_figure * temp
        rx_power_tensor[rx_power_tensor == 0] = -np.infty
        rx_power_tensor = 10 ** (0.1 * rx_power_tensor)  # in Watts

        self.snrs = rx_power_tensor / noise_power
        self.snrs[self.snrs == 0.0] = 1e-25  # no paths to -250 dB SNR
        # print(np.sum((self.snrs == 1e-25)))

        axis = 2 if self.path_mode == "min_delay" else 3
        self.x = np.stack((azimuth_aoa_tensor, azimuth_aod_tensor, zenith_aoa_tensor, zenith_aod_tensor, rx_power_tensor, toa_tensor), axis=axis)
        self.y = np.reshape(true_cord_tensor, (self.tot_points, true_cord_tensor.shape[2]))
        self.y = self.y - np.min(self.y, axis=0)  # trick to improve learning stability with RELU activation

        print(f"The shape of the feature data is {self.x.shape}")
        print(f"The shape of the target data is: {self.y.shape}")
        nans = np.sum(np.isnan(self.x))  # check the data is correctly formatted
        if nans > 0:
            raise KeyError

        self.input_shape = None
        self.output_shape = None

    def _get_path_data(self, tensor_data):
        if self.path_mode == "min_delay":
            tensor_data = tensor_data[self.indices[0], self.indices[1], self.indices[2], self.arg_min]
            return np.reshape(tensor_data, (self.tot_points, tensor_data.shape[2]))

        elif self.path_mode == "all_paths":
            return np.reshape(tensor_data, (self.tot_points, tensor_data.shape[2], tensor_data.shape[3]))
        else:
            raise NotImplemented

    def _compute_indices(self, tensor_data):
        if self.path_mode == "min_delay":
            tensor_data[tensor_data == 0] = 1  # need to set toa=1 where there's no path
            self.arg_min = np.argmin(tensor_data, axis=3)
            self.indices = np.indices(self.arg_min.shape)
            self.toa_tensor_compare = tensor_data[self.indices[0], self.indices[1], self.indices[2], self.arg_min]
        elif self.path_mode == "all_paths":
            self.arg_min = None
            self.indices = None
        else:
            raise NotImplemented

    def get_datasets(self, split=0.5, dnn=True, scale=True, scaler=None, add_noise=False, get_full_data=False):
        x = self.x
        y = self.y
        if add_noise:
            # add Gaussian noise to the data 
            x = x + x / np.sqrt(self.snrs)[:, :, :, None] * np.random.normal(0, 1, size=x.shape)

        # split the data between training and test sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=split, shuffle=True, random_state=self.rnd_state)
        print(f"The shape of the training data is: {x_train.shape}")
        print(f"The shape of the testing data is: {x_test.shape}")

        if get_full_data:
            # return the train and test data without reshaping and scaling
            return (x_train, y_train), (x_test, y_test)

        if dnn:
            # DNN case
            if self.path_mode == "min_delay":
                # min delay case
                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
                x_test = np.reshape(x_test, (x_train.shape[0], x_test.shape[1]*x_test.shape[2]))
            else:
                # all paths case
                x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]*x_train.shape[2]*x_train.shape[3]))
                x_test = np.reshape(x_test, (x_train.shape[0], x_test.shape[1]*x_test.shape[2]*x_test.shape[3]))
        else:
            # CNN case
            # TODO: scale the data accordingly
            train_set = CustomDataSet(x_train, y_train)
            test_set = CustomDataSet(x_test, y_test)
            self.input_shape = [x_train.shape[1], x_train.shape[2]]
            self.output_shape = self.y.shape[1]
            return train_set, test_set

        print(f"new training shape: {x_train.shape}")
        x_train = scaler.fit_transform(x_train) if scale else x_train
        x_test = scaler.transform(x_test) if scale else x_test
        train_set = CustomDataSet(x_train, y_train)
        test_set = CustomDataSet(x_test, y_test)
        self.input_shape = x_train.shape[1]
        self.output_shape = self.y.shape[1]
        return train_set, test_set
