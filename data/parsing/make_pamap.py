import hydra
import os
import numpy as np
import glob

from pathlib import Path
from scipy import stats as s
from scipy import constants
from scipy.signal import butter, filtfilt
from tqdm import tqdm

def get_data_content(data_path):
    # read flash.dat to a list of lists
    datContent = [i.strip().split() for i in open(data_path).readlines()]
    datContent = np.array(datContent)

    label_idx = 1
    timestamp_idx = 0
    x_wrist_idx = 4
    y_wrist_idx = 5
    z_wrist_idx = 6
    x_chest_idx = 21
    y_chest_idx = 22
    z_chest_idx = 23
    x_ankle_idx = 38
    y_ankle_idx = 39
    z_ankle_idx = 40

    x_wrist_gyr_idx = 10
    y_wrist_gyr_idx = 11
    z_wrist_gyr_idx = 12
    x_chest_gyr_idx = 27
    y_chest_gyr_idx = 28
    z_chest_gyr_idx = 29
    x_ankle_gyr_idx = 44
    y_ankle_gyr_idx = 45
    z_ankle_gyr_idx = 46

    x_wrist_mag_idx = 13
    y_wrist_mag_idx = 14
    z_wrist_mag_idx = 15
    x_chest_mag_idx = 30
    y_chest_mag_idx = 31
    z_chest_mag_idx = 32
    x_ankle_mag_idx = 47
    y_ankle_mag_idx = 48
    z_ankle_mag_idx = 49

    index_to_keep = [timestamp_idx, label_idx, 
                     x_wrist_idx, y_wrist_idx, z_wrist_idx,
                     x_chest_idx, y_chest_idx, z_chest_idx,
                     x_ankle_idx, y_ankle_idx, z_ankle_idx,
                     x_wrist_gyr_idx, y_wrist_gyr_idx, z_wrist_gyr_idx,
                     x_chest_gyr_idx, y_chest_gyr_idx, z_chest_gyr_idx,
                     x_ankle_gyr_idx, y_ankle_gyr_idx, z_ankle_gyr_idx,
                     x_wrist_mag_idx, y_wrist_mag_idx, z_wrist_mag_idx,
                     x_chest_mag_idx, y_chest_mag_idx, z_chest_mag_idx,
                     x_ankle_mag_idx, y_ankle_mag_idx, z_ankle_mag_idx]
    # 3d +- 16 g

    datContent = datContent[:, index_to_keep]
    datContent = datContent.astype(float)
    datContent = datContent[~np.isnan(datContent).any(axis=1)]
    return datContent


def content2x_and_y(data_content, epoch_len=30, sample_rate=100, overlap=15):
    sample_count = int(np.floor(len(data_content) / (epoch_len * sample_rate)))

    sample_label_idx = 1
    sample_x_wrist_idx = 2
    sample_y_wrist_idx = 3
    sample_z_wrist_idx = 4
    sample_x_chest_idx = 5
    sample_y_chest_idx = 6
    sample_z_chest_idx = 7
    sample_x_ankle_idx = 8
    sample_y_ankle_idx = 9
    sample_z_ankle_idx = 10
    sample_x_wrist_gyr_idx = 11
    sample_y_wrist_gyr_idx = 12
    sample_z_wrist_gyr_idx = 13
    sample_x_chest_gyr_idx = 14
    sample_y_chest_gyr_idx = 15
    sample_z_chest_gyr_idx = 16
    sample_x_ankle_gyr_idx = 17
    sample_y_ankle_gyr_idx = 18
    sample_z_ankle_gyr_idx = 19
    sample_x_wrist_mag_idx = 20
    sample_y_wrist_mag_idx = 21
    sample_z_wrist_mag_idx = 22
    sample_x_chest_mag_idx = 23
    sample_y_chest_mag_idx = 24
    sample_z_chest_mag_idx = 25
    sample_x_ankle_mag_idx = 26
    sample_y_ankle_mag_idx = 27
    sample_z_ankle_mag_idx = 28

    sample_limit = sample_count * epoch_len * sample_rate
    data_content = data_content[:sample_limit, :]

    label = data_content[:, sample_label_idx]
    x_wrist = data_content[:, sample_x_wrist_idx]
    y_wrist = data_content[:, sample_y_wrist_idx]
    z_wrist = data_content[:, sample_z_wrist_idx]
    x_chest = data_content[:, sample_x_chest_idx]
    y_chest = data_content[:, sample_y_chest_idx]
    z_chest = data_content[:, sample_z_chest_idx]
    x_ankle = data_content[:, sample_x_ankle_idx]
    y_ankle = data_content[:, sample_y_ankle_idx]
    z_ankle = data_content[:, sample_z_ankle_idx]
    x_wrist_gyr = data_content[:, sample_x_wrist_gyr_idx]
    y_wrist_gyr = data_content[:, sample_y_wrist_gyr_idx]
    z_wrist_gyr = data_content[:, sample_z_wrist_gyr_idx]
    x_chest_gyr = data_content[:, sample_x_chest_gyr_idx]
    y_chest_gyr = data_content[:, sample_y_chest_gyr_idx]
    z_chest_gyr = data_content[:, sample_z_chest_gyr_idx]
    x_ankle_gyr = data_content[:, sample_x_ankle_gyr_idx]
    y_ankle_gyr = data_content[:, sample_y_ankle_gyr_idx]
    z_ankle_gyr = data_content[:, sample_z_ankle_gyr_idx]
    x_wrist_mag = data_content[:, sample_x_wrist_mag_idx]
    y_wrist_mag = data_content[:, sample_y_wrist_mag_idx]
    z_wrist_mag = data_content[:, sample_z_wrist_mag_idx]
    x_chest_mag = data_content[:, sample_x_chest_mag_idx]
    y_chest_mag = data_content[:, sample_y_chest_mag_idx]
    z_chest_mag = data_content[:, sample_z_chest_mag_idx]
    x_ankle_mag = data_content[:, sample_x_ankle_mag_idx]
    y_ankle_mag = data_content[:, sample_y_ankle_mag_idx]
    z_ankle_mag = data_content[:, sample_z_ankle_mag_idx]


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
    shifted_x_wrist = data_content[offset:end_idx:, sample_x_wrist_idx]
    shifted_y_wrist = data_content[offset:end_idx:, sample_y_wrist_idx]
    shifted_z_wrist = data_content[offset:end_idx:, sample_z_wrist_idx]
    shifted_x_chest = data_content[offset:end_idx:, sample_x_chest_idx]
    shifted_y_chest = data_content[offset:end_idx:, sample_y_chest_idx]
    shifted_z_chest = data_content[offset:end_idx:, sample_z_chest_idx]
    shifted_x_ankle = data_content[offset:end_idx:, sample_x_ankle_idx]
    shifted_y_ankle = data_content[offset:end_idx:, sample_y_ankle_idx]
    shifted_z_ankle = data_content[offset:end_idx:, sample_z_ankle_idx]
    shifted_x_wrist_gyr = data_content[offset:end_idx:, sample_x_wrist_gyr_idx]
    shifted_y_wrist_gyr = data_content[offset:end_idx:, sample_y_wrist_gyr_idx]
    shifted_z_wrist_gyr = data_content[offset:end_idx:, sample_z_wrist_gyr_idx]
    shifted_x_chest_gyr = data_content[offset:end_idx:, sample_x_chest_gyr_idx]
    shifted_y_chest_gyr = data_content[offset:end_idx:, sample_y_chest_gyr_idx]
    shifted_z_chest_gyr = data_content[offset:end_idx:, sample_z_chest_gyr_idx]
    shifted_x_ankle_gyr = data_content[offset:end_idx:, sample_x_ankle_gyr_idx]
    shifted_y_ankle_gyr = data_content[offset:end_idx:, sample_y_ankle_gyr_idx]
    shifted_z_ankle_gyr = data_content[offset:end_idx:, sample_z_ankle_gyr_idx]
    shifted_x_wrist_mag = data_content[offset:end_idx:, sample_x_wrist_mag_idx]
    shifted_y_wrist_mag = data_content[offset:end_idx:, sample_y_wrist_mag_idx]
    shifted_z_wrist_mag = data_content[offset:end_idx:, sample_z_wrist_mag_idx]
    shifted_x_chest_mag = data_content[offset:end_idx:, sample_x_chest_mag_idx]
    shifted_y_chest_mag = data_content[offset:end_idx:, sample_y_chest_mag_idx]
    shifted_z_chest_mag = data_content[offset:end_idx:, sample_z_chest_mag_idx]
    shifted_x_ankle_mag = data_content[offset:end_idx:, sample_x_ankle_mag_idx]
    shifted_y_ankle_mag = data_content[offset:end_idx:, sample_y_ankle_mag_idx]
    shifted_z_ankle_mag = data_content[offset:end_idx:, sample_z_ankle_mag_idx]
                                   


    shifted_label = shifted_label.reshape(-1, epoch_len * sample_rate)
    shifted_x_wrist = shifted_x_wrist.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_wrist = shifted_y_wrist.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_wrist = shifted_z_wrist.reshape(-1, epoch_len * sample_rate, 1)
    shifted_x_chest = shifted_x_chest.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_chest = shifted_y_chest.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_chest = shifted_z_chest.reshape(-1, epoch_len * sample_rate, 1)
    shifted_x_ankle = shifted_x_ankle.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_ankle = shifted_y_ankle.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_ankle = shifted_z_ankle.reshape(-1, epoch_len * sample_rate, 1)
    shifted_x_wrist_gyr = shifted_x_wrist_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_wrist_gyr = shifted_y_wrist_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_wrist_gyr = shifted_z_wrist_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_x_chest_gyr = shifted_x_chest_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_chest_gyr = shifted_y_chest_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_chest_gyr = shifted_z_chest_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_x_ankle_gyr = shifted_x_ankle_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_ankle_gyr = shifted_y_ankle_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_ankle_gyr = shifted_z_ankle_gyr.reshape(-1, epoch_len * sample_rate, 1)
    shifted_x_wrist_mag = shifted_x_wrist_mag.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_wrist_mag = shifted_y_wrist_mag.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_wrist_mag = shifted_z_wrist_mag.reshape(-1, epoch_len * sample_rate, 1)
    shifted_x_chest_mag = shifted_x_chest_mag.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_chest_mag = shifted_y_chest_mag.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_chest_mag = shifted_z_chest_mag.reshape(-1, epoch_len * sample_rate, 1)
    shifted_x_ankle_mag = shifted_x_ankle_mag.reshape(-1, epoch_len * sample_rate, 1)
    shifted_y_ankle_mag = shifted_y_ankle_mag.reshape(-1, epoch_len * sample_rate, 1)
    shifted_z_ankle_mag = shifted_z_ankle_mag.reshape(-1, epoch_len * sample_rate, 1)
    
    
    
    shifted_X = np.concatenate([shifted_x_wrist, shifted_y_wrist, shifted_z_wrist,
                                shifted_x_chest, shifted_y_chest, shifted_z_chest,
                                shifted_x_ankle, shifted_y_ankle, shifted_z_ankle,
                                shifted_x_wrist_gyr, shifted_y_wrist_gyr, shifted_z_wrist_gyr,
                                shifted_x_chest_gyr, shifted_y_chest_gyr, shifted_z_chest_gyr,
                                shifted_x_ankle_gyr, shifted_y_ankle_gyr, shifted_z_ankle_gyr,
                                shifted_x_wrist_mag, shifted_y_wrist_mag, shifted_z_wrist_mag,
                                shifted_x_chest_mag, shifted_y_chest_mag, shifted_z_chest_mag,
                                shifted_x_ankle_mag, shifted_y_ankle_mag, shifted_z_ankle_mag], axis=2)

    label = label.reshape(-1, epoch_len * sample_rate)
    x_wrist = x_wrist.reshape(-1, epoch_len * sample_rate, 1)
    y_wrist = y_wrist.reshape(-1, epoch_len * sample_rate, 1)
    z_wrist = z_wrist.reshape(-1, epoch_len * sample_rate, 1)
    x_chest = x_chest.reshape(-1, epoch_len * sample_rate, 1)
    y_chest = y_chest.reshape(-1, epoch_len * sample_rate, 1)
    z_chest = z_chest.reshape(-1, epoch_len * sample_rate, 1)
    x_ankle = x_ankle.reshape(-1, epoch_len * sample_rate, 1)
    y_ankle = y_ankle.reshape(-1, epoch_len * sample_rate, 1)
    z_ankle = z_ankle.reshape(-1, epoch_len * sample_rate, 1)
    x_wrist_gyr = x_wrist_gyr.reshape(-1, epoch_len * sample_rate, 1)
    y_wrist_gyr = y_wrist_gyr.reshape(-1, epoch_len * sample_rate, 1)
    z_wrist_gyr = z_wrist_gyr.reshape(-1, epoch_len * sample_rate, 1)
    x_chest_gyr = x_chest_gyr.reshape(-1, epoch_len * sample_rate, 1)
    y_chest_gyr = y_chest_gyr.reshape(-1, epoch_len * sample_rate, 1)
    z_chest_gyr = z_chest_gyr.reshape(-1, epoch_len * sample_rate, 1)
    x_ankle_gyr = x_ankle_gyr.reshape(-1, epoch_len * sample_rate, 1)
    y_ankle_gyr = y_ankle_gyr.reshape(-1, epoch_len * sample_rate, 1)
    z_ankle_gyr = z_ankle_gyr.reshape(-1, epoch_len * sample_rate, 1)
    x_wrist_mag = x_wrist_mag.reshape(-1, epoch_len * sample_rate, 1)
    y_wrist_mag = y_wrist_mag.reshape(-1, epoch_len * sample_rate, 1)
    z_wrist_mag = z_wrist_mag.reshape(-1, epoch_len * sample_rate, 1)
    x_chest_mag = x_chest_mag.reshape(-1, epoch_len * sample_rate, 1)
    y_chest_mag = y_chest_mag.reshape(-1, epoch_len * sample_rate, 1)
    z_chest_mag = z_chest_mag.reshape(-1, epoch_len * sample_rate, 1)
    x_ankle_mag = x_ankle_mag.reshape(-1, epoch_len * sample_rate, 1)
    y_ankle_mag = y_ankle_mag.reshape(-1, epoch_len * sample_rate, 1)
    z_ankle_mag = z_ankle_mag.reshape(-1, epoch_len * sample_rate, 1)
    X = np.concatenate([x_wrist, y_wrist, z_wrist,
                        x_chest, y_chest, z_chest,
                        x_ankle, y_ankle, z_ankle,
                        x_wrist_gyr, y_wrist_gyr, z_wrist_gyr,
                        x_chest_gyr, y_chest_gyr, z_chest_gyr,
                        x_ankle_gyr, y_ankle_gyr, z_ankle_gyr,
                        x_wrist_mag, y_wrist_mag, z_wrist_mag,
                        x_chest_mag, y_chest_mag, z_chest_mag,
                        x_ankle_mag, y_ankle_mag, z_ankle_mag], axis=2)

    X = np.concatenate([X, shifted_X])
    label = np.concatenate([label, shifted_label])
    return X, label


def clean_up_label(X, labels):
    # 1. remove rows with >50% zeros
    sample_count_per_row = labels.shape[1]

    rows2keep = np.ones(labels.shape[0], dtype=bool)
    transition_class = 0
    for i in range(labels.shape[0]):
        row = labels[i, :]
        if np.sum(row == transition_class) > 0.5 * sample_count_per_row:
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

    for file_path in tqdm(file_paths):
        subject_id = int(file_path.split("/")[-1][-7:-4])
        datContent = get_data_content(file_path)
        current_X, current_y = content2x_and_y(
            datContent, epoch_len=epoch_len, overlap=overlap
        )
        current_X, current_y = clean_up_label(current_X, current_y)
        ids = np.full(
            shape=len(current_y), fill_value=subject_id, dtype=np.int64
        )
        if len(X) == 0:
            X = current_X
            y = current_y
            pid = ids
        else:
            X = np.concatenate([X, current_X])
            y = np.concatenate([y, current_y])
            pid = np.concatenate([pid, ids])

    y = y.flatten()
    X = X / constants.g  # convert to unit of g
    clip_value = 3
    X = np.clip(X, -clip_value, clip_value)

    # Keep only 8 activities that all subjects perform
    y_filter = (
        (y == 1)
        | (y == 2)
        | (y == 3)
        | (y == 4)
        | (y == 12)
        | (y == 13)
        | (y == 16)
        | (y == 17)
    )
    X = X[y_filter]
    y = y[y_filter]
    pid = pid[y_filter]

    # Butterworth Filter
    cutoff = 20.0
    order = 6
    sample_rate = 100
    filtered_x = butterworth_filter(X, cutoff, sample_rate, order)

    np.save(X_path, filtered_x)
    np.save(y_path, y)
    np.save(pid_path, pid)


def get_write_paths(secs):
    current_file = Path(__file__)
    data_clean_path = current_file.parent.parent.parent / "data" / "clean"
    dataset_folder = data_clean_path / "pamap"
    os.makedirs(dataset_folder, exist_ok=True)
    data_root = dataset_folder / f"pamap_w{secs:02d}"
    os.makedirs(data_root, exist_ok=True)

    X_path = os.path.join(data_root, "X.npy")
    y_path = os.path.join(data_root, "Y.npy")
    pid_path = os.path.join(data_root, "pid.npy")

    return X_path, y_path, pid_path


@hydra.main(config_path="../../conf", config_name="main", version_base=None)
def main(cfg): 
    data_root = cfg.PAMAP

    data_path = data_root + "Protocol/"
    protocol_file_paths = glob.glob(data_path + "*.dat")
    data_path = data_root + "Optional/"
    optional_file_paths = glob.glob(data_path + "*.dat")
    file_paths = protocol_file_paths + optional_file_paths

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
