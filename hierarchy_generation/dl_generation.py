import numpy as np
import pandas as pd
import torch
import logging
import yaml
import wandb
import torch.nn as nn
import torch.nn.functional as F

from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, to_tree
from sklearn.metrics import accuracy_score
from torcheval.metrics.functional import multiclass_f1_score

from utils import get_labels, EarlyStopping, separate_X_sensorwise, get_train_test_idxs, setup_data, initialise_optimizer
from models.utils import Resnet_fusion_n_IMU, Resnet_ensemble_block, load_weights


def initialise_hierarchy(
                    cfg,
                    my_device, 
                    X_feats,
                    y,
                    groups,
                    fold
                    ):
    hier_output_path = f"{cfg.tree.HAC_root}/{wandb.run.id}"
    
    if cfg.model.name != "Hi_OSCAR":
        print(f"Hierarchy generation not required for model {cfg.model.name}")
        return hier_output_path

    cluster_classes_dl(
            X_feats, 
            y, 
            groups, 
            my_device, 
            hier_output_path, 
            fold,
            cfg
        )
    
    print(f"Hierarchy YAML saved to {hier_output_path}.yaml")

def initialise_optimizer(cfg, model, prototypes=None):
    if "learning_rate" in cfg.model:
        learning_rate = cfg.model.learning_rate
    else:
        learning_rate = cfg.evaluation.learning_rate

    if cfg.model.name == "euc_dist":
        optimizer = torch.optim.Adam([
                    {"params":model.parameters()},
                    {"params":prototypes},
                    ],
                    lr=learning_rate, 
                    amsgrad=True
                )
    elif "optimizer" not in cfg.model:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, amsgrad=True
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, amsgrad=True
        )
    return optimizer

class ClusteringModel(torch.nn.Module):
    def __init__(
            self,
            cfg,
            weights,
            load_sb_weights=True,
            my_device=None,
            ):
        super(ClusteringModel, self).__init__()
        self.num_classes=cfg.data.num_classes
        self.weight_path=cfg.evaluation.flip_net_path
        self.n_channels=cfg.data.channels_per_sensor
        self.sensor_separation=cfg.model.sensor_separation
        self.has_pressure = ("atm_pressure" in cfg.data.sensors)
        self.num_sensors = len(cfg.data.sensors)
        self.num_IMU = len(cfg.data.IMU_positions)
        self.num_ind_sensors = self.num_IMU * (self.num_sensors - 1 * self.has_pressure)
        self.weights = weights
        self.device = my_device
        self.window_size = cfg.evaluation.window_size

        self.models = nn.ModuleList()
        for _ in range(self.num_ind_sensors):
            model_i = Resnet_ensemble_block(
                window_size=self.window_size,
                n_channels=self.n_channels 
            )
            self.models.append(model_i)
        if load_sb_weights:
            load_weights(
                    weight_path=self.weight_path,
                    model=self.models,
                    sensor_separation=self.sensor_separation,
                    my_device=my_device,
                    num_IMU=self.num_ind_sensors
                )
        # if there is pressure, append model which accepts 1 channel (not pretrained)
        if self.has_pressure:
            model_atm = Resnet_ensemble_block(
                    window_size=self.window_size,
                    n_channels=1 
                )
            self.models.append(model_atm)
            self.num_ind_sensors += 1

        self.feat_extractor = Resnet_fusion_n_IMU(sensor_models=self.models,
                                                  num_sensors=self.num_ind_sensors,
                                                  num_classes=self.num_classes,
                                                  window_size=self.window_size)    

    def forward(self, x):
        logits, feats = self.feat_extractor(x)
        return logits, feats
    
    def compute_loss(self, output, target):
        loss = F.cross_entropy(output, target, weight=self.weights)
        return loss, 0.0, 0.0
    
    def predict(self, logits):
        _, preds = torch.max(logits, dim=1)
        return preds

def evaluate_model(model, val_loader, cfg):
    model.eval()
    val_loss, val_other, val_soft, all_preds, all_ys = 0.0, 0.0, 0.0, [], []

    with torch.no_grad():
        for i, (my_X, my_Y) in enumerate(val_loader):
            my_X = separate_X_sensorwise(cfg, my_X, model.module.device)
            true_y = my_Y.to(model.module.device, dtype=torch.long)

            logits, feats = model(my_X)
            loss, _, _ = model.module.compute_loss(logits, true_y)

            val_loss += loss.item()

            preds = model.module.predict(logits)
            all_preds.extend(preds.cpu().numpy())
            all_ys.extend(true_y.cpu().numpy())
            
            # Clean up tensors to prevent memory accumulation
            del my_X, true_y, logits, feats, loss, preds

    val_loss /= len(val_loader)
    val_other /= len(val_loader)
    val_soft /= len(val_loader)
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)
    val_f1 = multiclass_f1_score(torch.tensor(all_preds), torch.tensor(all_ys), num_classes=len(cfg.data.label2code)+1, average="macro")
    val_acc = accuracy_score(all_ys, all_preds)

    return val_loss, val_other, val_soft, val_acc, val_f1


def generate_features(X_feats, y, groups, my_device, fold, cfg):
    """
    Train my model on multiclass classification, then from the training set, 
    extract the cluster centroid of each class
    Use that as the feature vector for each class.
    """
    y, le, labels = get_labels(y)
    if isinstance(X_feats, pd.DataFrame):
        X_feats = X_feats.to_numpy()
    
    # Select specific subjects/activities for testing
    train_idx, test_id_idx, test_ood_idx = get_train_test_idxs(cfg, y, groups, fold)

    # Get loaders up
    train_loader, val_loader, test_id_loader, test_ood_loader, weights = setup_data(
        train_idx, test_id_idx, test_ood_idx, X_feats, y, groups, cfg, class_weights=cfg.model.weighted, my_device=my_device
    ) 

    early_stopping = EarlyStopping(
        patience=cfg.evaluation.patience, path=cfg.model_path, verbose=True
    )
    # Initialize model
    model = ClusteringModel(
            cfg,
            weights,
            my_device=my_device,
        )
    model = torch.nn.DataParallel(model)
    model = model.to(my_device, dtype=torch.float)

    # Get optimiser
    optimizer = initialise_optimizer(cfg, model)

    # Train model
    train_model(
        cfg,
        model,
        optimizer,
        train_loader,
        val_loader,
        early_stopping
        )

    # Load trained model
    model = ClusteringModel(
            cfg,
            weights,
            my_device=my_device,
        )
    model = torch.nn.DataParallel(model)
    model = model.to(my_device, dtype=torch.float)
    model.load_state_dict(torch.load(cfg.model_path))


    # Generate class features on training set
    unique_labels = np.unique(y)
    num_classes = len(unique_labels)
    # Initialize lists to collect features for each class
    class_features_list = [[] for _ in range(num_classes)]    
    feature_dim = None
    model.eval()
    with torch.no_grad():
        for i, (my_X, my_Y) in enumerate(train_loader):
            my_X = separate_X_sensorwise(cfg, my_X, model.module.device)
            true_y = my_Y.to(model.module.device, dtype=torch.long)
            logits, feats = model(my_X)
            
            # Store feature dimension from first batch
            if feature_dim is None:
                feature_dim = feats.shape[1]
            
            # Group features by class
            for j, label in enumerate(true_y.cpu().numpy()):
                class_idx = np.where(unique_labels == label)[0][0]
                class_features_list[class_idx].append(feats[j].cpu().numpy())
            
            # Clean up tensors after processing batch
            del my_X, true_y, logits, feats
            torch.cuda.empty_cache()
    
    # Compute centroids for each class
    class_features = np.zeros((num_classes, feature_dim))
    
    for i in range(num_classes):
        if class_features_list[i]:  # Check if there are samples for this class
            class_features[i] = np.mean(np.array(class_features_list[i]), axis=0)
        else:
            class_features[i] = np.array([None] * feature_dim)  # Handle case with no samples for a class
    
    print(f"Generated class feature centroids with shape: {class_features.shape}")

    return class_features

def train_model(cfg, model, optimizer, train_loader, val_loader, early_stopping):
    for epoch in range(5):
        model.train()
        train_losses = 0

        for i, (my_X, my_Y) in enumerate(train_loader):
            # Clear cache before processing
            torch.cuda.empty_cache()
            
            my_X = separate_X_sensorwise(cfg, my_X, model.module.device)
            true_y = my_Y.to(model.module.device, dtype=torch.long)

            # Calculate loss 
            logits, feats = model(my_X)
            loss, loss_other, loss_soft = model.module.compute_loss(logits, true_y)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Clear gradients immediately after step
            for param in model.parameters():
                param.grad = None

            train_losses += loss.item()
            
            # Explicit cleanup to prevent memory accumulation
            del my_X, true_y, logits, feats, loss, loss_other, loss_soft
            torch.cuda.empty_cache()
            
        train_losses /= len(train_loader)

        val_loss, val_other, val_soft, val_acc, val_f1 = evaluate_model(model, val_loader, cfg)

        # print + log validation results
        print_msg = (
            f"[{epoch}/{4}] "
            + f"train_loss: {train_losses:.4e} "
            + f"val_loss: {val_loss:.4e} "
            + f"val_f1: {val_f1:.4f} " 
            + f"val_acc: {val_acc:.4f} " 
            + f"val_other: {val_other:.4e} " 
            + f"val_soft: {val_soft:.4e} "
            )
        early_stopping(val_loss, model)
        print(print_msg)
        
        # Clear GPU cache at end of epoch
        torch.cuda.empty_cache()
        
        # Early stopping
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    return True

def generate_hierarchy_yaml(Z, unique_labels, code2label, output_path="hierarchy.yaml"):
    # Create a tree from the linkage matrix
    root_node, nodes = to_tree(Z, rd=True)

    # Assign arbitrary names to internal nodes
    internal_node_counter = 0

    def get_internal_node_name():
        nonlocal internal_node_counter
        internal_node_counter += 1
        return f"Node {chr(65 + internal_node_counter - 1)}"

    def build_hierarchy_tree(node, node_index=None):
        if node.is_leaf():
            # Leaf node corresponds to a class
            label_index = node.id
            class_name = code2label[unique_labels[label_index]]
            return {
                "name": class_name,
                "type": "class",
                "Z_index": node_index,
                "class_id": int(unique_labels[label_index])
            }
        else:
            # Internal node corresponds to a group
            children = []
            if node.left is not None:
                children.append(build_hierarchy_tree(node.left, node.left.id))
            if node.right is not None:
                children.append(build_hierarchy_tree(node.right, node.right.id))
            return {
                "name": get_internal_node_name(),
                "type": "group",
                "Z_index": node_index,
                "children": children
            }

    # Build the tree starting from the root
    hierarchy = {
        "hierarchy": {
            "name": "Root",
            "type": "group",
            "Z_index": len(Z) + len(unique_labels) - 1,
            "children": build_hierarchy_tree(root_node, len(Z) + len(unique_labels) - 1)["children"]
        }
    }

    # Save the hierarchy to a YAML file
    with open(output_path, "w") as yaml_file:
        yaml.dump(hierarchy, yaml_file, default_flow_style=False, sort_keys=False)


def cluster_classes_dl(X_feats, y, groups, my_device, output_path, fold=[1], cfg=None):
    code2label = {code: label for label, code in cfg.data.label2code.items()}
    # Step 1: Prepare Labels
    y, le, labels = get_labels(y)
    unique_labels = np.unique(y)

    # Step 2: Generate Cluster centroids
    class_features = generate_features(X_feats, y, groups, my_device, fold, cfg)

    # Step 3: Handle missing classes
    # Identify which classes have features
    valid_class_mask = np.array([not np.all(np.isnan(features)) for features in class_features])
    valid_indices = np.where(valid_class_mask)[0]
    print(f"Valid classes: {len(valid_indices)} out of {len(class_features)}")
    # Create a mapping from original indices to valid indices
    valid_class_features = class_features[valid_class_mask]
    
    # Step 4: Scale the Aggregated Features (only valid ones)
    scaler = StandardScaler()
    scaled_valid_features = scaler.fit_transform(valid_class_features)
    scaled_valid_features = np.nan_to_num(scaled_valid_features)
    
    Z = linkage(scaled_valid_features, method='average', metric='cosine', optimal_ordering=True)
    np.save(output_path+"_Z.npy", Z)
    
    # Map unique_labels to only include valid classes
    unique_labels = unique_labels[valid_class_mask]

    # Step 8: Generate YAML hierarchy
    generate_hierarchy_yaml(Z, unique_labels, code2label, output_path+".yaml")

