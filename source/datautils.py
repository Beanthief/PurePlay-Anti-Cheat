import numpy
import torch

class WindowDataset(torch.utils.data.Dataset):
    def __init__(self, windows):
        self.windows = torch.tensor(windows, dtype=torch.float32)

    def __len__(self):
        return self.windows.size(0)

    def __getitem__(self, index):
        window = self.windows[index]
        return window, window

def fit_scaler(data):
    data_array = numpy.array(data)
    data_min = data_array.min(axis=0)
    data_max = data_array.max(axis=0)
    return data_min, data_max

def apply_scaler(data, data_min, data_max, feature_range=(0, 1)):
    data_array = numpy.array(data)
    data_range = data_max - data_min
    denominator = numpy.where(data_range == 0, 1, data_range)
    scale_value = (feature_range[1] - feature_range[0]) / denominator
    min_value = feature_range[0] - data_min * scale_value
    return data_array * scale_value + min_value