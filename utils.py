import numpy as np
import pandas as pd
import torch
import wandb
import collections
import random

from sklearn.model_selection import GroupShuffleSplit
from sklearn import preprocessing
from scipy.interpolate import interp1d
from transforms3d.axangles import axangle2mat
from torchvision import transforms
from torch.utils.data import DataLoader

from models.Hi_OSCAR import Hi_OSCAR

def initialise_model(cfg, 
                     my_device, 
                     weights, 
                     load_sb_weights=True,
                     hier_path=None
                     ):
    if cfg.model.name == "Hi_OSCAR":
        model = Hi_OSCAR(
            cfg,
            weights,
            load_sb_weights=load_sb_weights,
            my_device=my_device,
            hier_path=hier_path,
        )
    else:
        raise NotImplementedError(f"Model {cfg.model.name} not implemented")
    
    model = torch.nn.DataParallel(model)
    model = model.to(my_device, dtype=torch.float)
    return model

def initialise_optimizer(cfg, model):
    if "learning_rate" in cfg.model:
        learning_rate = cfg.model.learning_rate
    else:
        learning_rate = cfg.evaluation.learning_rate

    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, amsgrad=True
    )
    return optimizer


def load_from_npy(cfg):
    
    x_path = cfg.data.X_path
    y_path = cfg.data.Y_path
    pid_path = cfg.data.PID_path

    X = np.load(x_path)
    Y = np.load(y_path)
    P = np.load(pid_path)

    if cfg.data.dataset_name == "PAMAP2_w10":
        if set(cfg.data.sensors) == set(["accelerometer", "gyroscope", "magnetometer"]):
            if set(cfg.data.IMU_positions) == set(["wrist", "chest", "ankle"]):
                idx = np.arange(27)
            elif set(cfg.data.IMU_positions) == set(["wrist", "chest"]):
                idx = list(range(0, 6))+list(range(9, 15))+list(range(18,23))
            elif set(cfg.data.IMU_positions) == set(["wrist", "ankle"]):
                idx = list(range(0, 3))+list(range(6, 12))+list(range(15,21))+list(range(24,27))
            elif set(cfg.data.IMU_positions) == set(["chest", "ankle"]):
                idx = list(range(3, 9))+list(range(12, 18))+list(range(21,27))
            elif set(cfg.data.IMU_positions) == set(["wrist"]):
                idx = list(range(0, 3))+list(range(9, 12))+list(range(18,21))
            elif set(cfg.data.IMU_positions) == set(["chest"]):
                idx = list(range(3, 6))+list(range(12, 15))+list(range(21,24))
            elif set(cfg.data.IMU_positions) == set(["ankle"]):
                idx = list(range(6, 9))+list(range(15, 18))+list(range(24,27))
            
        elif set(cfg.data.sensors) == set(["accelerometer", "gyroscope"]):
            if set(cfg.data.IMU_positions) == set(["wrist", "chest", "ankle"]):
                idx = np.arange(18)
            elif set(cfg.data.IMU_positions) == set(["wrist", "chest"]):
                idx = list(range(0, 6))+list(range(9, 15))
            elif set(cfg.data.IMU_positions) == set(["wrist", "ankle"]):
                idx = list(range(0, 3))+list(range(6, 12))+list(range(15,18))
            elif set(cfg.data.IMU_positions) == set(["chest", "ankle"]):
                idx = list(range(3, 9))+list(range(12, 18))
            elif set(cfg.data.IMU_positions) == set(["wrist"]):
                idx = list(range(0, 3))+list(range(9, 12))
            elif set(cfg.data.IMU_positions) == set(["chest"]):
                idx = list(range(3, 6))+list(range(12, 15))
            elif set(cfg.data.IMU_positions) == set(["ankle"]):
                idx = list(range(6, 9))+list(range(15, 18))

        elif set(cfg.data.sensors) == set(["gyroscope", "magnetometer"]):
            if set(cfg.data.IMU_positions) == set(["wrist", "chest", "ankle"]):
                idx = np.arange(9,27)
            elif set(cfg.data.IMU_positions) == set(["wrist", "chest"]):
                idx = list(range(9, 15))+list(range(18,23))
            elif set(cfg.data.IMU_positions) == set(["wrist", "ankle"]):
                idx = list(range(9, 12))+list(range(15,21))+list(range(24,27))
            elif set(cfg.data.IMU_positions) == set(["chest", "ankle"]):
                idx = list(range(12, 18))+list(range(21,27))
            elif set(cfg.data.IMU_positions) == set(["wrist"]):
                idx = list(range(9, 12))+list(range(18,21))
            elif set(cfg.data.IMU_positions) == set(["chest"]):
                idx = list(range(12, 15))+list(range(21,24))
            elif set(cfg.data.IMU_positions) == set(["ankle"]):
                idx = list(range(15, 18))+list(range(24,27))

        elif set(cfg.data.sensors) == set(["accelerometer", "magnetometer"]):
            if set(cfg.data.IMU_positions) == set(["wrist", "chest", "ankle"]):
                idx = list(range(0,9))+list(range(18,27))
            elif set(cfg.data.IMU_positions) == set(["wrist", "chest"]):
                idx = list(range(0, 6))+list(range(18,23))
            elif set(cfg.data.IMU_positions) == set(["wrist", "ankle"]):
                idx = list(range(0, 3))+list(range(6,9))+list(range(18,21))+list(range(24,27))
            elif set(cfg.data.IMU_positions) == set(["chest", "ankle"]):
                idx = list(range(3, 9))+list(range(21,27))
            elif set(cfg.data.IMU_positions) == set(["wrist"]):
                idx = list(range(0, 3))+list(range(18,21))
            elif set(cfg.data.IMU_positions) == set(["chest"]):
                idx = list(range(3, 6))+list(range(21,24))
            elif set(cfg.data.IMU_positions) == set(["ankle"]):
                idx = list(range(6, 9))+list(range(24,27))

        elif set(cfg.data.sensors) == set(["accelerometer"]):
            if set(cfg.data.IMU_positions) == set(["wrist", "chest", "ankle"]):
                idx = np.arange(9)
            elif set(cfg.data.IMU_positions) == set(["wrist", "chest"]):
                idx = list(range(0, 6))
            elif set(cfg.data.IMU_positions) == set(["wrist", "ankle"]):
                idx = list(range(0, 3))+list(range(6, 9))
            elif set(cfg.data.IMU_positions) == set(["chest", "ankle"]):
                idx = list(range(3, 9))
            elif set(cfg.data.IMU_positions) == set(["wrist"]):
                idx = list(range(0, 3))
            elif set(cfg.data.IMU_positions) == set(["chest"]):
                idx = list(range(3, 6))
            elif set(cfg.data.IMU_positions) == set(["ankle"]):
                idx = list(range(6, 9))

        elif set(cfg.data.sensors) == set(["gyroscope"]):
            if set(cfg.data.IMU_positions) == set(["wrist", "chest", "ankle"]):
                idx = list(range(9,18))
            elif set(cfg.data.IMU_positions) == set(["wrist", "chest"]):
                idx = list(range(9, 15))
            elif set(cfg.data.IMU_positions) == set(["wrist", "ankle"]):
                idx = list(range(9, 12))+list(range(15,18))
            elif set(cfg.data.IMU_positions) == set(["chest", "ankle"]):
                idx = list(range(12, 18))
            elif set(cfg.data.IMU_positions) == set(["wrist"]):
                idx = list(range(9, 12))
            elif set(cfg.data.IMU_positions) == set(["chest"]):
                idx = list(range(12, 15))
            elif set(cfg.data.IMU_positions) == set(["ankle"]):
                idx = list(range(15, 18))

        elif set(cfg.data.sensors) == set(["magnetometer"]):
            if set(cfg.data.IMU_positions) == set(["wrist", "chest", "ankle"]):
                idx = list(range(18,27))
            elif set(cfg.data.IMU_positions) == set(["wrist", "chest"]):
                idx = list(range(18,23))
            elif set(cfg.data.IMU_positions) == set(["wrist", "ankle"]):
                idx = list(range(18,21))+list(range(24,27))
            elif set(cfg.data.IMU_positions) == set(["chest", "ankle"]):
                idx = list(range(21,27))
            elif set(cfg.data.IMU_positions) == set(["wrist"]):
                idx = list(range(18,21))
            elif set(cfg.data.IMU_positions) == set(["chest"]):
                idx = list(range(21,24))
            elif set(cfg.data.IMU_positions) == set(["ankle"]):
                idx = list(range(24,27))

    elif cfg.data.dataset_name == "oppo_w10" or cfg.data.dataset_name == "oppolctm_w10":
        if set(cfg.data.sensors) == set(["accelerometer", "gyroscope", "magnetometer"]):
            if set(cfg.data.IMU_positions) == set(["back", "RUA", "RLA", "LUA", "LLA"]):
                idx = np.arange(45)
            elif set(cfg.data.IMU_positions) == set(["back", "RUA", "RLA", "LUA"]):
                idx = list(range(0, 12))+list(range(15, 27))+list(range(30,42))
            elif set(cfg.data.IMU_positions) == set(["back", "RUA", "RLA", "LLA"]):
                idx = list(range(0, 9))+list(range(12, 24))+list(range(27,39))+list(range(42,45))
            elif set(cfg.data.IMU_positions) == set(["back", "RUA", "LUA", "LLA"]):
                idx = list(range(0, 6))+list(range(9, 21))+list(range(24,36))+list(range(39,45))
            elif set(cfg.data.IMU_positions) == set(["back", "RLA", "LUA", "LLA"]):
                idx = list(range(0, 3))+list(range(6, 18))+list(range(21,33))+list(range(36,45))
            elif set(cfg.data.IMU_positions) == set(["RUA", "RLA", "LUA", "LLA"]):
                idx = list(range(3, 15))+list(range(18, 30))+list(range(33,45))
            elif set(cfg.data.IMU_positions) == set(["back", "RUA", "RLA"]):
                idx = list(range(0, 9))+list(range(15, 24))+list(range(30,39))
            elif set(cfg.data.IMU_positions) == set(["back", "RUA", "LUA"]):
                idx = list(range(0, 6))+list(range(9, 12))+list(range(15,21))+list(range(24,27))+list(range(30,36))+list(range(39,42))
            elif set(cfg.data.IMU_positions) == set(["back", "RUA", "LLA"]):
                idx = list(range(0, 6))+list(range(12, 15))+list(range(15,21))+list(range(27,30))+list(range(30,36))+list(range(42,45))
            elif set(cfg.data.IMU_positions) == set(["back", "RLA", "LLA"]):
                idx = list(range(0, 3))+list(range(6, 9))+list(range(12,15))+list(range(15,18))+list(range(21,24))+list(range(27,30))+list(range(30,33))+list(range(36,39))+list(range(42,45))
            elif set(cfg.data.IMU_positions) == set(["back", "RLA", "LUA"]):
                idx = list(range(0, 3))+list(range(6, 9))+list(range(9,12))+list(range(15,18))+list(range(21,24))+list(range(24,27))+list(range(30,33))+list(range(36,39))+list(range(39,42))
            elif set(cfg.data.IMU_positions) == set(["back", "LUA", "LLA"]):
                idx = list(range(0, 3))+list(range(9, 12))+list(range(12,15))+list(range(15,18))+list(range(24,27))+list(range(27,30))+list(range(30,33))+list(range(39,42))+list(range(42,45))
            elif set(cfg.data.IMU_positions) == set(["RUA", "RLA", "LUA"]):
                idx = list(range(3, 6))+list(range(6, 9))+list(range(9,12))+list(range(18,21))+list(range(21,24))+list(range(24,27))+list(range(33,36))+list(range(36,39))+list(range(39,42))
            elif set(cfg.data.IMU_positions) == set(["RUA", "LUA", "LLA"]):
                idx = list(range(3, 6))+list(range(9, 12))+list(range(12,15))+list(range(18,21))+list(range(24,27))+list(range(27,30))+list(range(33,36))+list(range(39,42))+list(range(42,45))
            elif set(cfg.data.IMU_positions) == set(["RUA", "RLA", "LLA"]):
                idx = list(range(3, 6))+list(range(6, 9))+list(range(12,15))+list(range(18,21))+list(range(21,24))+list(range(27,30))+list(range(33,36))+list(range(36,39))+list(range(42,45))
            elif set(cfg.data.IMU_positions) == set(["RLA", "LUA", "LLA"]):
                idx = list(range(6, 9))+list(range(9, 12))+list(range(12,15))+list(range(21,24))+list(range(24,27))+list(range(27,30))+list(range(36,39))+list(range(39,42))+list(range(42,45))
            elif set(cfg.data.IMU_positions) == set(["back", "RUA"]):
                idx = list(range(0, 3))+list(range(3, 6))+list(range(15,18))+list(range(18,21))+list(range(30,33))+list(range(33,36))
            elif set(cfg.data.IMU_positions) == set(["back", "RLA"]):
                idx = list(range(0, 3))+list(range(6, 9))+list(range(15,18))+list(range(21,24))+list(range(30,33))+list(range(36,39))
            elif set(cfg.data.IMU_positions) == set(["back", "LUA"]):
                idx = list(range(0, 3))+list(range(9, 12))+list(range(15,18))+list(range(24,27))+list(range(30,33))+list(range(39,42))
            elif set(cfg.data.IMU_positions) == set(["back", "LLA"]):
                idx = list(range(0, 3))+list(range(12,15))+list(range(15,18))+list(range(27,30))+list(range(30,33))+list(range(42,45))
            elif set(cfg.data.IMU_positions) == set(["RUA", "RLA"]):
                idx = list(range(3, 6))+list(range(6, 9))+list(range(18,21))+list(range(21,24))+list(range(33,36))+list(range(36,39))
            elif set(cfg.data.IMU_positions) == set(["RUA", "LUA"]):
                idx = list(range(3, 6))+list(range(9, 12))+list(range(18,21))+list(range(24,27))+list(range(33,36))+list(range(39,42))
            elif set(cfg.data.IMU_positions) == set(["RUA", "LLA"]):
                idx = list(range(3, 6))+list(range(12,15))+list(range(18,21))+list(range(27,30))+list(range(33,36))+list(range(42,45))
            elif set(cfg.data.IMU_positions) == set(["RLA", "LUA"]):
                idx = list(range(6, 9))+list(range(9, 12))+list(range(21,24))+list(range(24,27))+list(range(36,39))+list(range(39,42))
            elif set(cfg.data.IMU_positions) == set(["RLA", "LLA"]):
                idx = list(range(6, 9))+list(range(12,15))+list(range(21,24))+list(range(27,30))+list(range(36,39))+list(range(42,45))
            elif set(cfg.data.IMU_positions) == set(["LUA", "LLA"]):
                idx = list(range(9, 12))+list(range(12,15))+list(range(24,27))+list(range(27,30))+list(range(39,42))+list(range(42,45))
            elif set(cfg.data.IMU_positions) == set(["back"]):
                idx = list(range(0, 3))+list(range(15,18))+list(range(30,33))
            elif set(cfg.data.IMU_positions) == set(["RUA"]):
                idx = list(range(3, 6))+list(range(18,21))+list(range(33,36))
            elif set(cfg.data.IMU_positions) == set(["RLA"]):
                idx = list(range(6, 9))+list(range(21,24))+list(range(36,39))
            elif set(cfg.data.IMU_positions) == set(["LUA"]):
                idx = list(range(9, 12))+list(range(24,27))+list(range(39,42))
            elif set(cfg.data.IMU_positions) == set(["LLA"]):
                idx = list(range(12,15))+list(range(27,30))+list(range(42,45))

    elif cfg.data.dataset_name == "NFI_HAR":
        if len(cfg.data.IMU_positions) == 2 and len(cfg.data.sensors) == 3:
            idx = np.arange(13)
        elif set(cfg.data.IMU_positions) == set(["back"]):
            print("using only back sensor")
            idx = list(range(0,6)) + list([12])
        elif set(cfg.data.IMU_positions) == set(["arm"]):
            print("using only arm sensor")
            idx = list(range(6,13))
        else:
            print(cfg.data.IMU_positions, cfg.data.sensors)
            print("Not implemented, using all data")
            idx = np.arange(13)
    
    elif cfg.data.dataset_name == "NFI_HARmix" or cfg.data.dataset_name == "NFI_HAR_ood":
        # Just select all sensors for now
        idx = np.arange(13)

    elif cfg.data.dataset_name == "realworld":
        idx = np.arange(45)

    X = X[:,:,idx]

    # Map Y values according to dictionary if the mapping exists in config
    if hasattr(cfg.data, 'old2new_code') and cfg.data.old2new_code:
        mapping_dict = cfg.data.old2new_code       
        # Apply the mapping
        Y = np.array([mapping_dict.get(int(y), y) for y in Y])
        print("Y values mapped according to old2new_code dictionary")

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)
    print("\nLabel distribution:")
    print(pd.Series(Y).value_counts())

    return X, Y, P

def get_labels(y):
    le = None
    labels = None
    # if cfg.data.task_type == "classify":
    le = preprocessing.LabelEncoder()
    labels = np.unique(y)
    le.fit(y)
    y = le.transform(y)
    
    return y, le, labels

def train_val_split(X, Y, group, val_size=0.125):
    num_split = 1
    folds = GroupShuffleSplit(
        num_split, test_size=val_size, random_state=41
    ).split(X, Y, groups=group)
    train_idx, val_idx = next(folds)
    return X[train_idx], X[val_idx], Y[train_idx], Y[val_idx]

def get_class_weights(y,cfg):
    # obtain inverse of frequency as weights for the loss function
    counter = collections.Counter(y)
    # Because OOD can truncate the end if the last activities are omitted
    if "NFI" in cfg.data.dataset_name:
        count_len = len(cfg.data.label2code)
    elif "oppo" in cfg.data.dataset_name:
        count_len = len(cfg.data.label2code)
    else:
        count_len = len(counter) 

    for i in range(count_len):
        if i not in counter.keys():
            counter[i] = 1
    num_samples = len(y)
    weights = [0] * len(counter)
    for idx in counter.keys():
        weights[idx] = 1.0 / (counter[idx] / num_samples)
    print("Weight tensor: ")
    print(weights)
    return weights

class NormalDataset:
    def __init__(
        self,
        X,
        y=[],
        pid=[],
        name="",
        isLabel=False,
        transform=None,
        target_transform=None,
        has_pressure=False,
    ):
        """
        Y needs to be in one-hot encoding
        X needs to be in N * Width
        Args:
            data_path (string): path to data
            files_to_load (list): subject names
            currently all npz format should allow support multiple ext

        """

        self.X = torch.from_numpy(X)
        self.y = y
        self.isLabel = isLabel
        self.transform = transform
        self.targetTransform = target_transform
        self.pid = pid
        self.has_pressure = has_pressure
        print(name + " set sample count : " + str(len(self.X)))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.X[idx, :]
        y = []
        if self.isLabel:
            y = self.y[idx]
            if self.targetTransform:
                y = self.targetTransform(y)

        if self.transform:
                sample = self.transform(sample)
        if len(self.pid) >= 1:
            return sample, y, self.pid[idx]
        else:
            return sample, y

def setup_data(train_idxs, test_id_idxs, test_ood_idxs, X_feats, Y, groups, cfg, class_weights=True, my_device="cpu"):
    tmp_X_train, X_id_test, X_ood_test = X_feats[train_idxs], X_feats[test_id_idxs], X_feats[test_ood_idxs]
    le = preprocessing.LabelEncoder()
    le.fit(Y[train_idxs])
    tmp_Y_train, Y_id_test, Y_ood_test = Y[train_idxs], Y[test_id_idxs], Y[test_ood_idxs]
    group_train, group_id_test, group_ood_test = groups[train_idxs], groups[test_id_idxs], groups[test_ood_idxs]

    X_train, X_val, Y_train, Y_val = train_val_split(  
                tmp_X_train, tmp_Y_train, group_train
            )
    
    # augmentation transforms
    my_transform = transforms.Compose([RandomSwitchAxis_nsensors(has_pressure=("atm_pressure" in cfg.data.sensors)), RotationAxis_nsensors(has_pressure=("atm_pressure" in cfg.data.sensors))])

    train_dataset = NormalDataset(
        X_train, Y_train, name="train", isLabel=True, transform=my_transform
    )
    val_dataset = NormalDataset(
        X_val, Y_val, name="val", isLabel=True
    )
    test_id_dataset = NormalDataset(
        X_id_test, Y_id_test, pid=group_id_test, name="test_id", isLabel=True
    )
    test_ood_dataset = NormalDataset(
        X_ood_test, Y_ood_test, pid=group_ood_test, name="test_ood", isLabel=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size= cfg.data.batch_size,
        shuffle=True,
        num_workers= cfg.evaluation.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size= cfg.data.batch_size,
        num_workers= cfg.evaluation.num_workers,
    )
    test_id_loader = DataLoader(
        test_id_dataset,
        batch_size= cfg.data.batch_size,
        num_workers= cfg.evaluation.num_workers,
    )
    test_ood_loader = DataLoader(
        test_ood_dataset,
        batch_size= cfg.data.batch_size,
        num_workers= cfg.evaluation.num_workers,
    )

    weights = None
    if class_weights:
        weights = get_class_weights(Y_train,cfg)
        weights = torch.tensor(weights).to(my_device)
    return train_loader, val_loader, test_id_loader, test_ood_loader, weights

def resize(X, length, axis=1):
    """Resize the temporal length using linear interpolation.
    X must be of shape (N,M,C) (channels last) or (N,C,M) (channels first),
    where N is the batch size, M is the temporal length, and C is the number
    of channels.
    If X is channels-last, use axis=1 (default).
    If X is channels-first, use axis=2.
    """
    length_orig = X.shape[axis]
    t_orig = np.linspace(0, 1, length_orig, endpoint=True)
    t_new = np.linspace(0, 1, length, endpoint=True)
    X = interp1d(t_orig, X, kind="linear", axis=axis, assume_sorted=True)(
        t_new
    )
    return X

def downsample_X(X, window_size):
    if X.shape[1] == window_size:
        print("No need to downsample")
        X_downsampled = X
    else:
        X_downsampled = resize(X, window_size)
    X_downsampled = X_downsampled.astype(
        "f4"
    )
    
    X_downsampled = np.transpose(X_downsampled, (0, 2, 1))
    print("X transformed shape:", X_downsampled.shape)
    
    return X_downsampled

class EarlyStopping:
    """Early stops the training if validation loss
    doesn't improve after a given patience."""

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pt",
        path_2=None,
        trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time v
                            alidation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each
                            validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity
                            to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.path_2 = path_2
        self.trace_func = trace_func

    def __call__(self, val_loss, model, prototypes=None):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, prototypes)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, prototypes)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, prototypes):
        """Saves model when validation loss decrease."""
        if self.verbose:
            msg = "Validation loss decreased"
            msg = msg + f"({self.val_loss_min:.6f} --> {val_loss:.6f})"
            msg = msg + "Saving model ..."
            self.trace_func(msg)
        torch.save(model.state_dict(), self.path)
        if self.path_2 is not None:
            torch.save(prototypes, self.path_2)
        self.val_loss_min = val_loss

def separate_X_sensorwise(cfg, my_X, my_device="cuda:0"):
    # Move input tensor to device first
    my_X = my_X.to(my_device, dtype=torch.float)
    
    if cfg.model.sensor_separation:
        if "atm_pressure" in cfg.data.sensors:
            num_sensors = len(cfg.data.IMU_positions)*(len(cfg.data.sensors)-1)
            my_Xs = []
            for i in range(num_sensors):
                start,end = i*3, (i+1)*3
                my_Xi = my_X[:,start:end,:].clone()  # Clone to avoid sharing memory
                my_Xs.append(my_Xi)
            my_x_atm = my_X[:,-1,:].unsqueeze(1).clone()  # Clone to avoid sharing memory
            my_Xs.append(my_x_atm)
        else:
            num_sensors = len(cfg.data.IMU_positions)*len(cfg.data.sensors)
            my_Xs = []
            for i in range(num_sensors):
                start,end = i*3, (i+1)*3
                my_Xi = my_X[:,start:end,:].clone()  # Clone to avoid sharing memory
                my_Xs.append(my_Xi)
    elif cfg.model.device_separation:
        if "atm_pressure" in cfg.data.sensors:
            num_channels = [3 * (len(cfg.data.sensors) - 1) + 1] + [3 * (len(cfg.data.sensors) - 1)] * (len(cfg.data.IMU_positions) - 1)
        else: 
            num_channels = [3 * len(cfg.data.sensors)] * len(cfg.data.IMU_positions)
        my_Xs = []
        start = 0
        for channels in num_channels:
            my_Xi = my_X[:,start:start+channels,:].clone()  # Clone to avoid sharing memory
            my_Xs.append(my_Xi)
            start += channels
    else:
        my_Xs = my_X
    
    return my_Xs

def get_train_test_idxs(cfg, y, groups, fold):
    test_subjects = fold

    # OOD analysis only done for NFI and Oppo datasets
    if "NFI" in cfg.data.dataset_name:
        ood_activities = list(cfg.tree.excluded_classes[cfg.tree.exclusion_choice].values())
    elif "oppo" in cfg.data.dataset_name:
        ood_activities = [max(y)] # The NULL class
    else: # no OOD activities for other datasets
        ood_activities = [] 

    test_id_idx = np.where(np.isin(groups, test_subjects) & ~(np.isin(y, ood_activities)))[0]
    test_ood_idx = np.where(np.isin(groups, test_subjects) & np.isin(y, ood_activities))[0]
    train_idx = np.where(~(np.isin(groups, test_subjects)) & ~(np.isin(y, ood_activities)))[0]

    return train_idx, test_id_idx, test_ood_idx


class RotationAxis_nsensors:
    def __init__(self, has_pressure):
        self.has_pressure = has_pressure
    def __call__(self, sample):
        # Extract number of sensors
        num_sensors = sample.shape[0] // 3 
        # Remove air pressure for later 
        if self.has_pressure:
            atm_pressure = sample[-1,:].unsqueeze(1)
            sample = sample[:-1,:]
        # Extract the sets of axes
        axes = sample.transpose(1,0)
        axes = np.split(axes, num_sensors, axis=1)
        axes = np.stack(axes, axis=0)

        # Rotate each set of axes independently
        for i in range(axes.shape[0]):
            sample_i = axes[i,:,:]
            angle = np.random.uniform(low=-np.pi, high=np.pi)
            axis = np.random.uniform(low=-1, high=1, size=sample_i.shape[1])
            sample_i = np.matmul(sample_i, axangle2mat(axis, angle))
            axes[i,:,:] = sample_i

        # Reshape the axes back to the original shape
        axes = np.concatenate(axes, axis=1)
        # Add back in pressure
        if self.has_pressure:
            axes = np.concatenate([axes,atm_pressure], axis=1)
        axes = np.swapaxes(axes, 0,1)
        axes = torch.tensor(axes)
        
        return axes

class RandomSwitchAxis_nsensors(object):
    """
    Randomly switch the three axises for the raw files
    Input size: 3 * FEATURE_SIZE
    """
    def __init__(self, has_pressure):
        self.has_pressure = has_pressure

    def __call__(self, sample):
        num_sensors = sample.shape[0] // 3
        choice = random.randint(1, 6)
        # Remove air pressure for later 
        if self.has_pressure:
            atm_pressure = sample[-1,:].unsqueeze(0)
            sample = sample[:-1,:]

        mix_list = []

        for i in range(num_sensors):
            # 3 * FEATURE
            idx = 3*i
            x = sample[idx, :]
            y = sample[idx+1, :]
            z = sample[idx+2, :]

            if choice == 1:
                mix_list.extend([x,y,z])
            elif choice == 2:
                mix_list.extend([x,z,y])
            elif choice == 3:
                mix_list.extend([y,x,z])
            elif choice == 4:
                mix_list.extend([y,z,x])
            elif choice == 5:
                mix_list.extend([z,x,y])
            elif choice == 6:
                mix_list.extend([z,y,x])

        sample = torch.stack(mix_list, dim=0)
        # Add back in pressure
        if self.has_pressure:
            sample = torch.cat([sample, atm_pressure], dim=0)
        
        return sample
