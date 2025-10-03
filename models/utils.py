import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class Resnet_ensemble_block(nn.Module):
    r"""The general form of the architecture can be described as follows:

    x->[Conv-[ResBlock]^m-BN-ReLU-Down]^n->y

    In other words:

            bn-relu-conv-bn-relu-conv                        bn-
           /                         \                      /
    x->conv --------------------------(+)-bn-relu-down-> conv ----

    """

    def __init__(
        self,
        window_size=10,
        n_channels=3,
    ):
        super(Resnet_ensemble_block, self).__init__()

        if window_size == 10:
            cgf = [
                (64, 5, 2, 5, 2, 2),
                (128, 5, 2, 5, 2, 2),
                (256, 5, 2, 5, 5, 1),
                (512, 5, 2, 5, 5, 1),
                (1024, 5, 0, 5, 3, 1),
            ]
        elif window_size == 5:
            cgf = [
                (64, 5, 2, 5, 2, 2),
                (128, 5, 2, 5, 2, 2),
                (256, 5, 2, 5, 3, 1),
                (256, 5, 2, 5, 3, 1),
                (512, 5, 0, 5, 3, 1),
            ]
        elif window_size == 30:
            cgf = [
                (64, 5, 2, 5, 3, 1),
                (128, 5, 2, 5, 3, 1),
                (256, 5, 2, 5, 5, 1),
                (512, 5, 2, 5, 5, 1),
                (1024, 5, 0, 5, 4, 0)
            ]

        in_channels = n_channels
        feature_extractor = nn.Sequential()
        for i, layer_params in enumerate(cgf):
            (
                out_channels,
                conv_kernel_size,
                n_resblocks,
                resblock_kernel_size,
                downfactor,
                downorder,
            ) = layer_params
            feature_extractor.add_module(
                f"layer{i+1}",
                Resnet_ensemble_block.make_layer(
                    in_channels,
                    out_channels,
                    conv_kernel_size,
                    n_resblocks,
                    resblock_kernel_size,
                    downfactor,
                    downorder,
                ),
            )
            in_channels = out_channels

        self.feature_extractor = feature_extractor

        weight_init(self)

    @staticmethod
    def make_layer(
        in_channels,
        out_channels,
        conv_kernel_size,
        n_resblocks,
        resblock_kernel_size,
        downfactor,
        downorder=1,
    ):
        r""" Basic layer in Resnets:

        x->[Conv-[ResBlock]^m-BN-ReLU-Down]->

        In other words:

                bn-relu-conv-bn-relu-conv
               /                         \
        x->conv --------------------------(+)-bn-relu-down->

        """

        # Check kernel sizes make sense (only odd numbers are supported)
        assert (
            conv_kernel_size % 2
        ), "Only odd number for conv_kernel_size supported"
        assert (
            resblock_kernel_size % 2
        ), "Only odd number for resblock_kernel_size supported"

        # Figure out correct paddings
        conv_padding = int((conv_kernel_size - 1) / 2)
        resblock_padding = int((resblock_kernel_size - 1) / 2)

        modules = [
            nn.Conv1d(
                in_channels,
                out_channels,
                conv_kernel_size,
                1,
                conv_padding,
                bias=False,
                padding_mode="circular",
            )
        ]

        for i in range(n_resblocks):
            modules.append(
                ResBlock(
                    out_channels,
                    out_channels,
                    resblock_kernel_size,
                    1,
                    resblock_padding,
                )
            )

        modules.append(nn.BatchNorm1d(out_channels))
        modules.append(nn.ReLU(True))
        modules.append(Downsample(out_channels, downfactor, downorder))

        return nn.Sequential(*modules)

    def forward(self, x):
        feats = self.feature_extractor(x)
        # print("DEBUG feats:", feats.size())

        return feats


class ResBlock(nn.Module):
    r""" Basic bulding block in Resnets:

       bn-relu-conv-bn-relu-conv
      /                         \
    x --------------------------(+)->

    """

    def __init__(
        self, in_channels, out_channels, kernel_size=5, stride=1, padding=2
    ):

        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(x))
        x = self.conv1(x)
        # print("DEBUG post conv1:", x.size())
        x = self.relu(self.bn2(x))
        x = self.conv2(x)

        x = x + identity

        return x

class Resnet_fusion_n_IMU(nn.Module):
    def __init__(self, 
                 sensor_models,
                 num_sensors, 
                 num_classes,
                 window_size):

        super(Resnet_fusion_n_IMU, self).__init__()
        self.sensor_models = nn.ModuleList(sensor_models)
        self.num_sensors = num_sensors
        self.sensor_output_size = (512 if window_size==5 else 1024) # defined by pretrained model

        # Dropout layer
        self.dropout = nn.Dropout(0.3)

        # Output layer 
        self.output_layer = nn.Linear(self.sensor_output_size*num_sensors, num_classes)

    def forward(self, inputs):
        # sensor_outputs should be a list of tensors with shape [batch_size, sensor_output_size]
        sensor_outputs = [model(x).squeeze(2) for model, x in zip(self.sensor_models, inputs)]

        # Concatenate outputs
        fusion_output = torch.cat(sensor_outputs, dim=1) # change to sum?
        fusion_output = self.dropout(fusion_output)

        # FC layer to connect to output classes
        output = self.output_layer(fusion_output)

        return output, fusion_output # return final output (output) and last layer embeddings (fusion_output)
    

class Downsample(nn.Module):
    r"""Downsampling layer that applies anti-aliasing filters.
    For example, order=0 corresponds to a box filter (or average downsampling
    -- this is the same as AvgPool in Pytorch), order=1 to a triangle filter
    (or linear downsampling), order=2 to cubic downsampling, and so on.
    See https://richzhang.github.io/antialiased-cnns/ for more details.
    """

    def __init__(self, channels=None, factor=2, order=1):
        super(Downsample, self).__init__()
        assert factor > 1, "Downsampling factor must be > 1"
        self.stride = factor
        self.channels = channels
        self.order = order

        # Figure out padding and check params make sense
        # The padding is given by order*(factor-1)/2
        # so order*(factor-1) must be divisible by 2
        total_padding = order * (factor - 1)
        assert total_padding % 2 == 0, (
            "Misspecified downsampling parameters."
            "Downsampling factor and order must be such "
            "that order*(factor-1) is divisible by 2"
        )
        self.padding = int(order * (factor - 1) / 2)

        box_kernel = np.ones(factor)
        kernel = np.ones(factor)
        for _ in range(order):
            kernel = np.convolve(kernel, box_kernel)
        kernel /= np.sum(kernel)
        kernel = torch.Tensor(kernel)
        self.register_buffer(
            "kernel", kernel[None, None, :].repeat((channels, 1, 1))
        )

    def forward(self, x):
        return F.conv1d(
            x,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.shape[1],
        )

def weight_init(self, mode="fan_out", nonlinearity="relu"):

    for m in self.modules():

        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.kaiming_normal_(
                m.weight, mode=mode, nonlinearity=nonlinearity
            )

        elif isinstance(m, (nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

def load_weights(
        weight_path, 
        model, 
        # model_name,
        sensor_separation,
        my_device, 
        num_IMU,
        has_pressure=False,
):
    num_sensors = num_IMU
    # only need to change weights name when
    # the model is trained in a distributed manner

    pretrained_dict = torch.load(weight_path, map_location=my_device)
    pretrained_dict_v2 = copy.deepcopy(
        pretrained_dict
    )  # v2 has the right para names

    # distributed pretraining can be inferred from the keys' module. prefix
    head = next(iter(pretrained_dict_v2)).split('.')[0]  # get head of first key
    if head == 'module':
        # remove module. prefix from dict keys
        pretrained_dict_v2 = {k.partition('module.')[2]: pretrained_dict_v2[k] for k in pretrained_dict_v2.keys()}


    # if model_name == "sensor_based" or model_name == "leto_bm" or model_name == "leto_sm":
    if sensor_separation:
        if hasattr(model[0], 'module'):
            model_dict = model[0].module.state_dict()
            multi_gpu_ft = True
        else:
            model_dict = model[0].state_dict()
            multi_gpu_ft = False

    elif has_pressure:
        # Get pretrained weight tensor (weight)
        old_weight = pretrained_dict_v2['feature_extractor.layer1.0.weight']
        # create new weight tensor with repeated channels
        new_weight = torch.cat([old_weight]*(num_sensors-has_pressure), dim=1)
        # Add a single channel for atm_pressure
        new_weight = torch.cat([new_weight, new_weight[:,0,:].unsqueeze(1)], dim=1)
        # reassign weight tensor with increased size
        pretrained_dict_v2['feature_extractor.layer1.0.weight'] = new_weight
        if hasattr(model, 'module'):
            model_dict = model.module.state_dict()
            multi_gpu_ft = True
        else:
            model_dict = model.state_dict()
            multi_gpu_ft = False

    # early fusion or no fusion (n_sensors == 1)
    else: 
        # Get pretrained weight tensor (weight)
        old_weight = pretrained_dict_v2['feature_extractor.layer1.0.weight']
        # create new weight tensor with repeated channels
        new_weight = torch.cat([old_weight]*num_sensors, dim=1)
        # reassign weight tensor with increased size
        pretrained_dict_v2['feature_extractor.layer1.0.weight'] = new_weight
        if hasattr(model, 'module'):
            model_dict = model.module.state_dict()
            multi_gpu_ft = True
        else:
            model_dict = model.state_dict()
            multi_gpu_ft = False


    # 1. filter out unnecessary keys such as the final linear layers
    #    we don't want linear layer weights either
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict_v2.items()
        if k in model_dict and k.split(".")[0] != "classifier"
    }

    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)

    # 3. load the new state dict
    # if model_name == "sensor_based" or model_name == "leto_bm" or model_name == "leto_sm":
    if sensor_separation:
        if multi_gpu_ft:
            for modeli in model:
                modeli.module.load_state_dict(model_dict) 
        else:
            for modeli in model:
                modeli.load_state_dict(model_dict)

    else:
        if multi_gpu_ft:
            model.module.load_state_dict(model_dict)
        else:
            model.load_state_dict(model_dict)

    print("%d Weights loaded" % len(pretrained_dict))
