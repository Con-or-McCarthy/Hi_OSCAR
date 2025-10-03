import numpy as np
import glob
import os
import hydra

from pathlib import Path
from tqdm import tqdm
from scipy.signal import butter, filtfilt
from scipy import stats as s



def get_data_content(data_path):
    # read flash.dat to a list of lists
    datContent = [i.strip().split() for i in open(data_path).readlines()]
    datContent = np.array(datContent)

    label_idx = 243 # 243 for locomotive activities, 244 for Higher-Level activities
    timestamp_idx = 0
    back_acc_x_idx = 37
    back_acc_y_idx = 38
    back_acc_z_idx = 39
    back_gyr_x_idx = 40
    back_gyr_y_idx = 41
    back_gyr_z_idx = 42
    back_mag_x_idx = 43
    back_mag_y_idx = 44
    back_mag_z_idx = 45

    rua_acc_x_idx = 50
    rua_acc_y_idx = 51
    rua_acc_z_idx = 52
    rua_gyr_x_idx = 53
    rua_gyr_y_idx = 54
    rua_gyr_z_idx = 55
    rua_mag_x_idx = 56
    rua_mag_y_idx = 57
    rua_mag_z_idx = 58

    rla_acc_x_idx = 63
    rla_acc_y_idx = 64
    rla_acc_z_idx = 65
    rla_gyr_x_idx = 66
    rla_gyr_y_idx = 67
    rla_gyr_z_idx = 68
    rla_mag_x_idx = 69
    rla_mag_y_idx = 70
    rla_mag_z_idx = 71

    lua_acc_x_idx = 76
    lua_acc_y_idx = 77
    lua_acc_z_idx = 78
    lua_gyr_x_idx = 79
    lua_gyr_y_idx = 80
    lua_gyr_z_idx = 81
    lua_mag_x_idx = 82
    lua_mag_y_idx = 83
    lua_mag_z_idx = 84

    lla_acc_x_idx = 89
    lla_acc_y_idx = 90
    lla_acc_z_idx = 91
    lla_gyr_x_idx = 92
    lla_gyr_y_idx = 93
    lla_gyr_z_idx = 94
    lla_mag_x_idx = 95
    lla_mag_y_idx = 96
    lla_mag_z_idx = 97
    

    index_to_keep = [timestamp_idx, label_idx, 
                     back_acc_x_idx, back_acc_y_idx, back_acc_z_idx, 
                     back_gyr_x_idx, back_gyr_y_idx, back_gyr_z_idx, 
                     back_mag_x_idx, back_mag_y_idx, back_mag_z_idx,
                     rua_acc_x_idx, rua_acc_y_idx, rua_acc_z_idx, 
                     rua_gyr_x_idx, rua_gyr_y_idx, rua_gyr_z_idx, 
                     rua_mag_x_idx, rua_mag_y_idx, rua_mag_z_idx,
                     rla_acc_x_idx, rla_acc_y_idx, rla_acc_z_idx,
                     rla_gyr_x_idx, rla_gyr_y_idx, rla_gyr_z_idx,
                     rla_mag_x_idx, rla_mag_y_idx, rla_mag_z_idx,
                     lua_acc_x_idx, lua_acc_y_idx, lua_acc_z_idx,
                     lua_gyr_x_idx, lua_gyr_y_idx, lua_gyr_z_idx,
                     lua_mag_x_idx, lua_mag_y_idx, lua_mag_z_idx,
                     lla_acc_x_idx, lla_acc_y_idx, lla_acc_z_idx,
                     lla_gyr_x_idx, lla_gyr_y_idx, lla_gyr_z_idx,
                     lla_mag_x_idx, lla_mag_y_idx, lla_mag_z_idx]
    # 3d +- 16 g

    datContent = datContent[:, index_to_keep]
    datContent = datContent.astype(float)
    datContent = datContent[~np.isnan(datContent).any(axis=1)]
    return datContent


def content2x_and_y(data_content, epoch_len=30, sample_rate=100, overlap=15):
    sample_count = int(np.floor(len(data_content) / (epoch_len * sample_rate)))

    sample_label_idx = 1
    sample_x_back_acc_idx = 2
    sample_y_back_acc_idx = 3
    sample_z_back_acc_idx = 4
    sample_x_back_gyr_idx = 5
    sample_y_back_gyr_idx = 6
    sample_z_back_gyr_idx = 7
    sample_x_back_mag_idx = 8
    sample_y_back_mag_idx = 9
    sample_z_back_mag_idx = 10
    sample_x_rua_acc_idx = 11
    sample_y_rua_acc_idx = 12
    sample_z_rua_acc_idx = 13
    sample_x_rua_gyr_idx = 14
    sample_y_rua_gyr_idx = 15
    sample_z_rua_gyr_idx = 16
    sample_x_rua_mag_idx = 17
    sample_y_rua_mag_idx = 18
    sample_z_rua_mag_idx = 19
    sample_x_rla_acc_idx = 20
    sample_y_rla_acc_idx = 21
    sample_z_rla_acc_idx = 22
    sample_x_rla_gyr_idx = 23
    sample_y_rla_gyr_idx = 24
    sample_z_rla_gyr_idx = 25
    sample_x_rla_mag_idx = 26
    sample_y_rla_mag_idx = 27
    sample_z_rla_mag_idx = 28
    sample_x_lua_acc_idx = 29
    sample_y_lua_acc_idx = 30
    sample_z_lua_acc_idx = 31
    sample_x_lua_gyr_idx = 32
    sample_y_lua_gyr_idx = 33
    sample_z_lua_gyr_idx = 34
    sample_x_lua_mag_idx = 35
    sample_y_lua_mag_idx = 36
    sample_z_lua_mag_idx = 37
    sample_x_lla_acc_idx = 38
    sample_y_lla_acc_idx = 39
    sample_z_lla_acc_idx = 40
    sample_x_lla_gyr_idx = 41
    sample_y_lla_gyr_idx = 42
    sample_z_lla_gyr_idx = 43
    sample_x_lla_mag_idx = 44
    sample_y_lla_mag_idx = 45
    sample_z_lla_mag_idx = 46


    sample_limit = sample_count * epoch_len * sample_rate
    data_content = data_content[:sample_limit, :]

    label = data_content[:, sample_label_idx]
    x_back_acc = data_content[:, sample_x_back_acc_idx]
    y_back_acc = data_content[:, sample_y_back_acc_idx]
    z_back_acc = data_content[:, sample_z_back_acc_idx]
    x_back_gyr = data_content[:, sample_x_back_gyr_idx]
    y_back_gyr = data_content[:, sample_y_back_gyr_idx]
    z_back_gyr = data_content[:, sample_z_back_gyr_idx]
    x_back_mag = data_content[:, sample_x_back_mag_idx]
    y_back_mag = data_content[:, sample_y_back_mag_idx]
    z_back_mag = data_content[:, sample_z_back_mag_idx]
    x_rua_acc = data_content[:, sample_x_rua_acc_idx]
    y_rua_acc = data_content[:, sample_y_rua_acc_idx]
    z_rua_acc = data_content[:, sample_z_rua_acc_idx]
    x_rua_gyr = data_content[:, sample_x_rua_gyr_idx]
    y_rua_gyr = data_content[:, sample_y_rua_gyr_idx]
    z_rua_gyr = data_content[:, sample_z_rua_gyr_idx]
    x_rua_mag = data_content[:, sample_x_rua_mag_idx]
    y_rua_mag = data_content[:, sample_y_rua_mag_idx]
    z_rua_mag = data_content[:, sample_z_rua_mag_idx]
    x_rla_acc = data_content[:, sample_x_rla_acc_idx]
    y_rla_acc = data_content[:, sample_y_rla_acc_idx]
    z_rla_acc = data_content[:, sample_z_rla_acc_idx]
    x_rla_gyr = data_content[:, sample_x_rla_gyr_idx]
    y_rla_gyr = data_content[:, sample_y_rla_gyr_idx]
    z_rla_gyr = data_content[:, sample_z_rla_gyr_idx]
    x_rla_mag = data_content[:, sample_x_rla_mag_idx]
    y_rla_mag = data_content[:, sample_y_rla_mag_idx]
    z_rla_mag = data_content[:, sample_z_rla_mag_idx]
    x_lua_acc = data_content[:, sample_x_lua_acc_idx]
    y_lua_acc = data_content[:, sample_y_lua_acc_idx]
    z_lua_acc = data_content[:, sample_z_lua_acc_idx]
    x_lua_gyr = data_content[:, sample_x_lua_gyr_idx]
    y_lua_gyr = data_content[:, sample_y_lua_gyr_idx]
    z_lua_gyr = data_content[:, sample_z_lua_gyr_idx]
    x_lua_mag = data_content[:, sample_x_lua_mag_idx]
    y_lua_mag = data_content[:, sample_y_lua_mag_idx]
    z_lua_mag = data_content[:, sample_z_lua_mag_idx]
    x_lla_acc = data_content[:, sample_x_lla_acc_idx]
    y_lla_acc = data_content[:, sample_y_lla_acc_idx]
    z_lla_acc = data_content[:, sample_z_lla_acc_idx]
    x_lla_gyr = data_content[:, sample_x_lla_gyr_idx]
    y_lla_gyr = data_content[:, sample_y_lla_gyr_idx]
    z_lla_gyr = data_content[:, sample_z_lla_gyr_idx]
    x_lla_mag = data_content[:, sample_x_lla_mag_idx]
    y_lla_mag = data_content[:, sample_y_lla_mag_idx]
    z_lla_mag = data_content[:, sample_z_lla_mag_idx]


    # to make overlappting window
    offset = overlap * sample_rate
    
    # Calculate how much data we have after applying offset
    shifted_data_length = len(data_content) - 2 * offset
    window_size = epoch_len * sample_rate
    
    # Ensure the shifted data length is divisible by window_size
    shifted_sample_count = shifted_data_length // window_size
    adjusted_shifted_length = shifted_sample_count * window_size
    
    # Extract shifted data with proper length
    end_idx = offset + adjusted_shifted_length
    shifted_label = data_content[offset:end_idx, sample_label_idx]
    shifted_x_back_acc = data_content[offset:end_idx, sample_x_back_acc_idx]
    shifted_y_back_acc = data_content[offset:end_idx, sample_y_back_acc_idx]
    shifted_z_back_acc = data_content[offset:end_idx, sample_z_back_acc_idx]
    shifted_x_back_gyr = data_content[offset:end_idx, sample_x_back_gyr_idx]
    shifted_y_back_gyr = data_content[offset:end_idx, sample_y_back_gyr_idx]
    shifted_z_back_gyr = data_content[offset:end_idx, sample_z_back_gyr_idx]
    shifted_x_back_mag = data_content[offset:end_idx, sample_x_back_mag_idx]
    shifted_y_back_mag = data_content[offset:end_idx, sample_y_back_mag_idx]
    shifted_z_back_mag = data_content[offset:end_idx, sample_z_back_mag_idx]
    shifted_x_rua_acc = data_content[offset:end_idx, sample_x_rua_acc_idx]
    shifted_y_rua_acc = data_content[offset:end_idx, sample_y_rua_acc_idx]
    shifted_z_rua_acc = data_content[offset:end_idx, sample_z_rua_acc_idx]
    shifted_x_rua_gyr = data_content[offset:end_idx, sample_x_rua_gyr_idx]
    shifted_y_rua_gyr = data_content[offset:end_idx, sample_y_rua_gyr_idx]
    shifted_z_rua_gyr = data_content[offset:end_idx, sample_z_rua_gyr_idx]
    shifted_x_rua_mag = data_content[offset:end_idx, sample_x_rua_mag_idx]
    shifted_y_rua_mag = data_content[offset:end_idx, sample_y_rua_mag_idx]
    shifted_z_rua_mag = data_content[offset:end_idx, sample_z_rua_mag_idx]
    shifted_x_rla_acc = data_content[offset:end_idx, sample_x_rla_acc_idx]
    shifted_y_rla_acc = data_content[offset:end_idx, sample_y_rla_acc_idx]
    shifted_z_rla_acc = data_content[offset:end_idx, sample_z_rla_acc_idx]
    shifted_x_rla_gyr = data_content[offset:end_idx, sample_x_rla_gyr_idx]
    shifted_y_rla_gyr = data_content[offset:end_idx, sample_y_rla_gyr_idx]
    shifted_z_rla_gyr = data_content[offset:end_idx, sample_z_rla_gyr_idx]
    shifted_x_rla_mag = data_content[offset:end_idx, sample_x_rla_mag_idx]
    shifted_y_rla_mag = data_content[offset:end_idx, sample_y_rla_mag_idx]
    shifted_z_rla_mag = data_content[offset:end_idx, sample_z_rla_mag_idx]
    shifted_x_lua_acc = data_content[offset:end_idx, sample_x_lua_acc_idx]
    shifted_y_lua_acc = data_content[offset:end_idx, sample_y_lua_acc_idx]
    shifted_z_lua_acc = data_content[offset:end_idx, sample_z_lua_acc_idx]
    shifted_x_lua_gyr = data_content[offset:end_idx, sample_x_lua_gyr_idx]
    shifted_y_lua_gyr = data_content[offset:end_idx, sample_y_lua_gyr_idx]
    shifted_z_lua_gyr = data_content[offset:end_idx, sample_z_lua_gyr_idx]
    shifted_x_lua_mag = data_content[offset:end_idx, sample_x_lua_mag_idx]
    shifted_y_lua_mag = data_content[offset:end_idx, sample_y_lua_mag_idx]
    shifted_z_lua_mag = data_content[offset:end_idx, sample_z_lua_mag_idx]
    shifted_x_lla_acc = data_content[offset:end_idx, sample_x_lla_acc_idx]
    shifted_y_lla_acc = data_content[offset:end_idx, sample_y_lla_acc_idx]
    shifted_z_lla_acc = data_content[offset:end_idx, sample_z_lla_acc_idx]
    shifted_x_lla_gyr = data_content[offset:end_idx, sample_x_lla_gyr_idx]
    shifted_y_lla_gyr = data_content[offset:end_idx, sample_y_lla_gyr_idx]
    shifted_z_lla_gyr = data_content[offset:end_idx, sample_z_lla_gyr_idx]
    shifted_x_lla_mag = data_content[offset:end_idx, sample_x_lla_mag_idx]
    shifted_y_lla_mag = data_content[offset:end_idx, sample_y_lla_mag_idx]
    shifted_z_lla_mag = data_content[offset:end_idx, sample_z_lla_mag_idx]


    shifted_label = shifted_label.reshape(-1, epoch_len * sample_rate)
    shifted_x_back_acc = shifted_x_back_acc.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_back_acc = shifted_y_back_acc.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_back_acc = shifted_z_back_acc.reshape(-1, epoch_len * sample_rate, 1)
    shifted_x_back_gyr = shifted_x_back_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_back_gyr = shifted_y_back_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_back_gyr = shifted_z_back_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_x_back_mag = shifted_x_back_mag.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_back_mag = shifted_y_back_mag.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_back_mag = shifted_z_back_mag.reshape(-1, epoch_len * sample_rate, 1)
    shifted_x_rua_acc = shifted_x_rua_acc.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_rua_acc = shifted_y_rua_acc.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_rua_acc = shifted_z_rua_acc.reshape(-1, epoch_len * sample_rate, 1)
    shifted_x_rua_gyr = shifted_x_rua_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_rua_gyr = shifted_y_rua_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_rua_gyr = shifted_z_rua_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_x_rua_mag = shifted_x_rua_mag.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_rua_mag = shifted_y_rua_mag.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_rua_mag = shifted_z_rua_mag.reshape(-1, epoch_len * sample_rate, 1)
    shifted_x_rla_acc = shifted_x_rla_acc.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_rla_acc = shifted_y_rla_acc.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_rla_acc = shifted_z_rla_acc.reshape(-1, epoch_len * sample_rate, 1)
    shifted_x_rla_gyr = shifted_x_rla_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_rla_gyr = shifted_y_rla_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_rla_gyr = shifted_z_rla_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_x_rla_mag = shifted_x_rla_mag.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_rla_mag = shifted_y_rla_mag.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_rla_mag = shifted_z_rla_mag.reshape(-1, epoch_len * sample_rate, 1)
    shifted_x_lua_acc = shifted_x_lua_acc.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_lua_acc = shifted_y_lua_acc.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_lua_acc = shifted_z_lua_acc.reshape(-1, epoch_len * sample_rate, 1)
    shifted_x_lua_gyr = shifted_x_lua_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_lua_gyr = shifted_y_lua_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_lua_gyr = shifted_z_lua_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_x_lua_mag = shifted_x_lua_mag.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_lua_mag = shifted_y_lua_mag.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_lua_mag = shifted_z_lua_mag.reshape(-1, epoch_len * sample_rate, 1)
    shifted_x_lla_acc = shifted_x_lla_acc.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_lla_acc = shifted_y_lla_acc.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_lla_acc = shifted_z_lla_acc.reshape(-1, epoch_len * sample_rate, 1)
    shifted_x_lla_gyr = shifted_x_lla_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_lla_gyr = shifted_y_lla_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_lla_gyr = shifted_z_lla_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_x_lla_mag = shifted_x_lla_mag.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_lla_mag = shifted_y_lla_mag.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_lla_mag = shifted_z_lla_mag.reshape(-1, epoch_len * sample_rate, 1)


    shifted_X = np.concatenate([shifted_x_back_acc, shifted_y_back_acc, shifted_z_back_acc,
                                shifted_x_rua_acc, shifted_y_rua_acc, shifted_z_rua_acc,
                                shifted_x_rla_acc, shifted_y_rla_acc, shifted_z_rla_acc,
                                shifted_x_lua_acc, shifted_y_lua_acc, shifted_z_lua_acc,
                                shifted_x_lla_acc, shifted_y_lla_acc, shifted_z_lla_acc,
                                shifted_x_back_gyr, shifted_y_back_gyr, shifted_z_back_gyr,
                                shifted_x_rua_gyr, shifted_y_rua_gyr, shifted_z_rua_gyr,
                                shifted_x_rla_gyr, shifted_y_rla_gyr, shifted_z_rla_gyr,
                                shifted_x_lua_gyr, shifted_y_lua_gyr, shifted_z_lua_gyr,
                                shifted_x_lla_gyr, shifted_y_lla_gyr, shifted_z_lla_gyr,
                                shifted_x_back_mag, shifted_y_back_mag, shifted_z_back_mag,
                                shifted_x_rua_mag, shifted_y_rua_mag, shifted_z_rua_mag,
                                shifted_x_rla_mag, shifted_y_rla_mag, shifted_z_rla_mag,
                                shifted_x_lua_mag, shifted_y_lua_mag, shifted_z_lua_mag,
                                shifted_x_lla_mag, shifted_y_lla_mag, shifted_z_lla_mag], axis=2)

    label = label.reshape(-1, epoch_len * sample_rate)
    x_back_acc = x_back_acc.reshape(-1, epoch_len * sample_rate, 1)
    y_back_acc = y_back_acc.reshape(-1, epoch_len * sample_rate, 1)
    z_back_acc = z_back_acc.reshape(-1, epoch_len * sample_rate, 1)
    x_back_gyr = x_back_gyr.reshape(-1, epoch_len * sample_rate, 1)
    y_back_gyr = y_back_gyr.reshape(-1, epoch_len * sample_rate, 1)
    z_back_gyr = z_back_gyr.reshape(-1, epoch_len * sample_rate, 1)
    x_back_mag = x_back_mag.reshape(-1, epoch_len * sample_rate, 1)
    y_back_mag = y_back_mag.reshape(-1, epoch_len * sample_rate, 1)
    z_back_mag = z_back_mag.reshape(-1, epoch_len * sample_rate, 1)
    x_rua_acc = x_rua_acc.reshape(-1, epoch_len * sample_rate, 1)
    y_rua_acc = y_rua_acc.reshape(-1, epoch_len * sample_rate, 1)
    z_rua_acc = z_rua_acc.reshape(-1, epoch_len * sample_rate, 1)
    x_rua_gyr = x_rua_gyr.reshape(-1, epoch_len * sample_rate, 1)
    y_rua_gyr = y_rua_gyr.reshape(-1, epoch_len * sample_rate, 1)
    z_rua_gyr = z_rua_gyr.reshape(-1, epoch_len * sample_rate, 1)
    x_rua_mag = x_rua_mag.reshape(-1, epoch_len * sample_rate, 1)
    y_rua_mag = y_rua_mag.reshape(-1, epoch_len * sample_rate, 1)
    z_rua_mag = z_rua_mag.reshape(-1, epoch_len * sample_rate, 1)
    x_rla_acc = x_rla_acc.reshape(-1, epoch_len * sample_rate, 1)
    y_rla_acc = y_rla_acc.reshape(-1, epoch_len * sample_rate, 1)
    z_rla_acc = z_rla_acc.reshape(-1, epoch_len * sample_rate, 1)
    x_rla_gyr = x_rla_gyr.reshape(-1, epoch_len * sample_rate, 1)
    y_rla_gyr = y_rla_gyr.reshape(-1, epoch_len * sample_rate, 1)
    z_rla_gyr = z_rla_gyr.reshape(-1, epoch_len * sample_rate, 1)
    x_rla_mag = x_rla_mag.reshape(-1, epoch_len * sample_rate, 1)
    y_rla_mag = y_rla_mag.reshape(-1, epoch_len * sample_rate, 1)
    z_rla_mag = z_rla_mag.reshape(-1, epoch_len * sample_rate, 1)
    x_lua_acc = x_lua_acc.reshape(-1, epoch_len * sample_rate, 1)
    y_lua_acc = y_lua_acc.reshape(-1, epoch_len * sample_rate, 1)
    z_lua_acc = z_lua_acc.reshape(-1, epoch_len * sample_rate, 1)
    x_lua_gyr = x_lua_gyr.reshape(-1, epoch_len * sample_rate, 1)
    y_lua_gyr = y_lua_gyr.reshape(-1, epoch_len * sample_rate, 1)
    z_lua_gyr = z_lua_gyr.reshape(-1, epoch_len * sample_rate, 1)
    x_lua_mag = x_lua_mag.reshape(-1, epoch_len * sample_rate, 1)
    y_lua_mag = y_lua_mag.reshape(-1, epoch_len * sample_rate, 1)
    z_lua_mag = z_lua_mag.reshape(-1, epoch_len * sample_rate, 1)
    x_lla_acc = x_lla_acc.reshape(-1, epoch_len * sample_rate, 1)
    y_lla_acc = y_lla_acc.reshape(-1, epoch_len * sample_rate, 1)
    z_lla_acc = z_lla_acc.reshape(-1, epoch_len * sample_rate, 1)
    x_lla_gyr = x_lla_gyr.reshape(-1, epoch_len * sample_rate, 1)
    y_lla_gyr = y_lla_gyr.reshape(-1, epoch_len * sample_rate, 1)
    z_lla_gyr = z_lla_gyr.reshape(-1, epoch_len * sample_rate, 1)
    x_lla_mag = x_lla_mag.reshape(-1, epoch_len * sample_rate, 1)
    y_lla_mag = y_lla_mag.reshape(-1, epoch_len * sample_rate, 1)
    z_lla_mag = z_lla_mag.reshape(-1, epoch_len * sample_rate, 1)

    X = np.concatenate([x_back_acc, y_back_acc, z_back_acc,
                        x_rua_acc, y_rua_acc, z_rua_acc,
                        x_rla_acc, y_rla_acc, z_rla_acc,
                        x_lua_acc, y_lua_acc, z_lua_acc,
                        x_lla_acc, y_lla_acc, z_lla_acc,
                        x_back_gyr, y_back_gyr, z_back_gyr,
                        x_rua_gyr, y_rua_gyr, z_rua_gyr,
                        x_rla_gyr, y_rla_gyr, z_rla_gyr,
                        x_lua_gyr, y_lua_gyr, z_lua_gyr,
                        x_lla_gyr, y_lla_gyr, z_lla_gyr,
                        x_back_mag, y_back_mag, z_back_mag,
                        x_rua_mag, y_rua_mag, z_rua_mag,
                        x_rla_mag, y_rla_mag, z_rla_mag,
                        x_lua_mag, y_lua_mag, z_lua_mag,
                        x_lla_mag, y_lla_mag, z_lla_mag], axis=2)

    X = np.concatenate([X, shifted_X])
    label = np.concatenate([label, shifted_label])
    return X, label


def clean_up_label(X, labels):
    # 1. remove rows with >50% zeros
    sample_count_per_row = labels.shape[1]

    rows2keep = np.ones(labels.shape[0], dtype=bool)
    for i in range(labels.shape[0]):
        row = labels[i, :]
        if np.sum(row == 0) > 0.5 * sample_count_per_row:
            rows2keep[i] = False

    labels = labels[rows2keep]
    X = X[rows2keep]

    # 2. majority voting for label in each epoch
    final_labels = []
    for i in range(labels.shape[0]):
        row = labels[i, :]
        final_labels.append(s.mode(row)[0])
    final_labels = np.array(final_labels, dtype=int)
    # print("Clean X shape: ", X.shape)
    # print("Clean y shape: ", final_labels.shape)
    return X, final_labels


def post_process_oppo(X, y, pid):
    # # Removing NULL class
    # zero_filter = np.array(y != 0)

    # X = X[zero_filter]
    # y = y[zero_filter]
    # pid = pid[zero_filter]

    # change NULL label to -1
    y[y==0] = -1  
    # change lie label from 5 to 3
    y[y == 5] = 3
    return X, y, pid

def butterworth_filter(data, cutoff, fs, order):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = filtfilt(b, a, data, axis=1)
    return filtered_data


def process_all(file_paths, X_path, y_path, pid_path, epoch_len, overlap):
    X = []
    y = []
    pid = []
    sample_rate = 33

    for file_path in tqdm(file_paths):
        # print(file_path)
        subject_id = int(file_path.split("/")[-1][1:2])

        datContent = get_data_content(file_path)
        current_X, current_y = content2x_and_y(
            datContent,
            sample_rate=sample_rate,
            epoch_len=epoch_len,
            overlap=overlap,
        )
        current_X, current_y = clean_up_label(current_X, current_y)
        if len(current_y) == 0:
            continue
        ids = np.full(
            shape=len(current_y), fill_value=subject_id, dtype=np.int32
        )
        if len(X) == 0:
            X = current_X
            y = current_y
            pid = ids
        else:
            X = np.concatenate([X, current_X])
            y = np.concatenate([y, current_y])
            pid = np.concatenate([pid, ids])

    # post-process
    y = y.flatten()
    X = X / 1000  # convert to g
    clip_value = 3
    X = np.clip(X, -clip_value, clip_value)
    X, y, pid = post_process_oppo(X, y, pid)

    # Butterworth Filter
    cutoff = 15.0
    order = 6
    # sample_rate = 33
    filtered_x = butterworth_filter(X, cutoff, sample_rate, order)

    np.save(X_path, filtered_x)
    np.save(y_path, y)
    np.save(pid_path, pid)


def get_write_paths(secs):
    current_file = Path(__file__)
    data_clean_path = current_file.parent.parent.parent / "data" / "clean"
    dataset_folder = data_clean_path / "opportunity"
    os.makedirs(dataset_folder, exist_ok=True)
    data_root = dataset_folder / f"oppo_lctm_w{secs:02d}"
    os.makedirs(data_root, exist_ok=True)

    X_path = os.path.join(data_root, "X.npy")
    y_path = os.path.join(data_root, "Y.npy")
    pid_path = os.path.join(data_root, "pid.npy")

    return X_path, y_path, pid_path


@hydra.main(config_path="../../conf", config_name="main", version_base=None)
def main(cfg): 
    data_root = cfg.OPPO
    data_path = data_root + "dataset/"
    file_paths = glob.glob(data_path + "*.dat")

    print("Processing for 10sec window..")
    X_path, y_path, pid_path = get_write_paths(10)
    epoch_len = 10
    overlap = 5
    process_all(file_paths, X_path, y_path, pid_path, epoch_len, overlap)
    print("Saved X to ", X_path)
    print("Saved y to ", y_path)

    print("Processing for 5sec window..")
    X_path, y_path, pid_path = get_write_paths(5)
    epoch_len = 5
    overlap = 2
    process_all(file_paths, X_path, y_path, pid_path, epoch_len, overlap)
    print("Saved X to ", X_path)
    print("Saved y to ", y_path)


if __name__ == "__main__":
    main()
