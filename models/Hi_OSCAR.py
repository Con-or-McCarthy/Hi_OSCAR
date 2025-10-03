import numpy as np
import yaml
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, List
from collections import defaultdict

from models.utils import Resnet_fusion_n_IMU, Resnet_ensemble_block, load_weights

class Hi_OSCAR(nn.Module):
    def __init__(
            self,
            cfg,
            weights,
            load_sb_weights=True,
            my_device=None,
            hier_path=None,
            ):
        super(Hi_OSCAR, self).__init__()
        # Define the number of IMUs, sensors, and classes etc.
        self.num_IMU = len(cfg.data.IMU_positions)
        self.has_pressure = ("atm_pressure" in cfg.data.sensors)
        self.num_sensors = len(cfg.data.sensors)
        self.num_ind_sensors = self.num_IMU * (self.num_sensors - 1 * self.has_pressure)
        self.n_channels=cfg.data.channels_per_sensor
        self.num_classes=cfg.data.num_classes
        self.sensor_separation=cfg.model.sensor_separation
        self.weight_path=cfg.evaluation.flip_net_path
        self.store_entropy = cfg.model.store_entropy
        self.window_size = cfg.evaluation.window_size

        # For inference stopping criterion:
        self.entropy_tracker = HierarchicalEntropyTracker()
        
        self.weights = weights
        self.device = my_device

        # Setting up hierarchy
        if hier_path is None:
            self.tree_path = f"{cfg.tree.HAC_root}/{wandb.run.id}.yaml"
            Z_path = f"{cfg.tree.HAC_root}/{wandb.run.id}_Z.npy"
        else:
            self.tree_path = f"{hier_path}.yaml"
            Z_path = f"{hier_path}_Z.npy"
        try:
            self.Z = np.load(Z_path)
            self.Z_transformed = self.transform_linkage_matrix()
        except:
            self.Z = None

        self.excluded_classes = cfg.tree.excluded_classes[cfg.tree.exclusion_choice]
        self.hierarchy_nodes = self.generate_hierarchy_nodes(self.tree_path, excluded_classes=self.excluded_classes.values())
        # self.num_output_nodes = len(self.hierarchy_nodes) # DEBUG
        self.paths = self.create_paths_from_hierarchy(self.hierarchy_nodes)
        self.use_hierarchy = True
        
        self.beta = cfg.model.beta
        self.euc_output_size = (512 if self.window_size == 5 else 1024) # Size of the output from the feature extractor
        self.embed_dims = cfg.model.embed_dims
        self.drop_p = cfg.model.drop_p
        
        # Create feature extractor
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
        
        # output nodes for different branches
        self.feat_ext_size = self.euc_output_size*len(self.models)
        # self.num_output_nodes = sum(1 for node in self.hierarchy_nodes if node["type"] == "class")

        self.fc = nn.Linear(self.feat_ext_size, self.feat_ext_size)
        self.dropout = nn.Dropout(p=self.drop_p)
        self.relu = nn.ReLU()
        self.out_layer = nn.Linear(
            in_features=self.feat_ext_size,
            out_features=self.num_output_nodes
        )

    def forward(self, x_in):
        _, feats = self.feat_extractor(x_in)
        x = self.out_layer(feats)
        return x, feats
    
    def loss_soft_multilevel(self, output, target):
        """
        Computes L_soft for a multi-level, unequal hierarchy.

        Args:
            output: Tensor of shape (batch_size, num_output_nodes) - raw logits from the network.
            target: Tensor of shape (batch_size,) - ground truth labels.
            paths: Dictionary with keys as class IDs and values as lists of nodes representing the path from root to that class.

        Returns:
            L_soft: Scalar tensor representing the soft loss.
        """
        batch_size = output.size(0)
        L_soft = 0

        paths = self.paths
        weights = self.weights
        device = self.device
        target_classes = target.tolist()

        for i, target_class in enumerate(target_classes):
            # Get the path from the root to the target class
            path = paths[target_class]

            # Calculate the cross-entropy loss for each node along the path
            for level, node in enumerate(path):
                # Extract the corresponding logit index for the current node
                if node["children"]:
                    # If the node has children, it's an internal node, and we need to select among its children
                    child_indices = [child["logit_index"] for child in node["children"]]
                    child_logits = output[i, child_indices]  # Extract the logits for all children of the current node

                    # Determine the correct child index based on the path
                    correct_child_node = path[level + 1] if (level + 1) < len(path) else node
                    correct_child_logit_index = correct_child_node["logit_index"]
                    target_idx = child_indices.index(correct_child_logit_index)

                    # Compute cross-entropy for the current level
                    loss = F.cross_entropy(
                        child_logits.unsqueeze(0), 
                        torch.tensor([target_idx], device=device)
                        )

                    # Apply weights if provided
                    if weights is not None:
                        loss *= weights[target_class]

                    L_soft += loss

        L_soft /= batch_size
        return L_soft
    
    def loss_other_multilevel(self, output, target):
        """
        Computes L_other for a multi-level, unequal hierarchy.

        Args:
            output: Tensor of shape (batch_size, num_output_nodes) - raw logits from the network.
            target: Tensor of shape (batch_size,) - ground truth labels.
            paths: Dictionary with keys as class IDs and values as lists of nodes representing the path from root to that class.
            hierarchy_nodes: List of all nodes (including their children) representing the entire hierarchy.

        Returns:
            L_other: Scalar tensor representing the other loss.
        """
        batch_size = output.shape[0]
        L_other = 0

        paths = self.paths
        device = output.device
        target_classes = target.tolist()

        for i, target_class in enumerate(target_classes):
            # Get the path from the root to the target class
            path = paths[target_class]  # Now a list of node dictionaries
            # Extract the indices of all nodes along the path
            path_node_indices = {node["logit_index"] for node in path}

            # Compute L_other for nodes not in the path of the ground-truth node
            for node in self.hierarchy_nodes:
                # If the node's logit index is not in the path, it should be driven to uniformity
                if node["logit_index"] not in path_node_indices and node["children"]:
                    # Get the indices of the children for the current node
                    child_indices = [child["logit_index"] for child in node["children"]]

                    # Extract the logits corresponding to the children of this node
                    child_log_probs = F.log_softmax(output[i, child_indices], dim=0)

                    # Create a uniform distribution target over the children
                    uniform_dist = torch.full(child_log_probs.shape, 1.0 / len(child_indices), device=device)

                    # Compute cross-entropy between the softmax output and the uniform distribution
                    L_other += F.kl_div(child_log_probs, uniform_dist, reduction='sum')

        L_other /= batch_size
        return L_other

    def compute_loss(self, output, target, feats=None):
        # remapped_target = torch.tensor([self.class_mapping[t.item()] for t in target], dtype=torch.long, device=target.device)
        if self.use_hierarchy:
            l_other = self.loss_other_multilevel(output, target)
            l_soft = self.loss_soft_multilevel(output, target)
            loss = l_soft + self.beta * l_other
        else:
            loss = F.cross_entropy(output, target, weight=self.weights)
            l_other = torch.tensor(0, device=output.device)
            l_soft = torch.tensor(0, device=output.device)

        return loss, l_other, l_soft
        
    def generate_hierarchy_nodes(self, hierarchy_yaml_path, excluded_classes=None):
        """
        Generates a hierarchy_nodes list from the YAML structure provided.

        Args:
            hierarchy_yaml_path: Path to the YAML file containing the hierarchy structure.
            excluded_classes: List of class names or class IDs to exclude from the hierarchy.

        Returns:
            hierarchy_nodes: A list of dictionaries, where each dictionary contains information about a node in the hierarchy.
        """
        if excluded_classes is None:
            excluded_classes = []

        # Load YAML data
        with open(hierarchy_yaml_path, 'r') as file:
            hierarchy_data = yaml.safe_load(file)

        hierarchy_nodes = []
        current_logit_index = 0  # Start index for output layer logits

        def traverse(node, level):
            nonlocal current_logit_index

            # Skip nodes that are excluded classes
            if node["type"] == "class" and node.get("class_id") in excluded_classes:
                return None

            # Create node dictionary
            node_dict = {
                "name": node["name"],
                "type": node["type"],
                "logit_index": node["Z_index"], # logit_indes == Z_index
                "children": [],
                "level": level,
                "class_id": node.get("class_id") if node["type"] == "class" else None
            }
            
            # Increment logit index for each node processed (internal or leaf)
            current_logit_index += 1

            # Process children recursively
            if "children" in node:
                for child in node["children"]:
                    child_node = traverse(child, level + 1)
                    if child_node is not None:
                        node_dict["children"].append(child_node)

            hierarchy_nodes.append(node_dict)
            return node_dict

        # Start traversal from the root node
        traverse(hierarchy_data["hierarchy"], 0)

        # Update total output nodes based on the number of nodes traversed
        self.num_output_nodes = current_logit_index # DEBUG

        return hierarchy_nodes
        
    def create_paths_from_hierarchy(self, hierarchy_nodes):
        """
        Creates paths from root to each leaf node in the hierarchy.

        Args:
            hierarchy_nodes: List of dictionaries representing nodes in the hierarchy, where each node contains
                            information like logit_index, name, type, children, etc.

        Returns:
            paths: Dictionary with keys as class IDs and values as lists of nodes (dictionaries) representing 
                the path from root to that class.
        """
        paths = {}

        def traverse(node, current_path):
            # Add the current node to the path
            new_path = current_path + [node]

            # If the node is a leaf node (i.e., a class), store the path
            if node["type"] == "class":
                class_id = node["class_id"]
                if class_id is not None:
                    paths[class_id] = new_path
            else:
                # If the node has children, continue traversing
                for child in node["children"]:
                    traverse(child, new_path)

        # Find root nodes and start traversal
        for node in hierarchy_nodes:
            if node["level"] == 0:  # Start traversal from root-level nodes
                traverse(node, [])

        return paths
    
    def cross_entropy(self, pred_probs, target_probs):
        epsilon = 1e-10
        pred_probs = pred_probs + epsilon

        # Compute the cross-entropy
        cross_entropy_loss = -torch.sum(target_probs * torch.log(pred_probs))
        
        return cross_entropy_loss

    def predict(self, output):
        """
        Predicts the class based on the maximum total path probability to a leaf node.
        All possible paths from root to each leaf are evaluated.

        Args:
            output: Tensor of shape (batch_size, num_output_nodes) - raw logits from the network.

        Returns:
            predicted_classes: Tensor of shape (batch_size,) representing the predicted class for each sample.
        """
        output = self.normalise_probs(output)
        if self.use_hierarchy:
            batch_size = output.size(0)
            predicted_classes = []

            for i in range(batch_size):
                max_path_prob = -float('inf')
                best_leaf_class = None

                for act, path in self.paths.items():
                    act_prob = 1.0
                    for node in path[1:]:
                        logit_index = node["logit_index"]
                        prob = output[i, logit_index].item()
                        act_prob *= prob
                    if act_prob > max_path_prob:
                        max_path_prob = act_prob
                        best_leaf_class = act

                predicted_classes.append(best_leaf_class)
            predicted_classes = torch.tensor(predicted_classes, dtype=torch.long, device=output.device)
        else:
            predicted_classes = torch.argmax(output, dim=1)

        return predicted_classes

    def normalise_probs(self, output):
        if self.use_hierarchy:
            # Apply softmax to the internal nodes as per the hierarchy
            softmax_probs = torch.zeros_like(output)
            for node in self.hierarchy_nodes:
                if node["type"] == "group" or node["children"]:
                    # Apply softmax to logits corresponding to the children of the current node
                    child_indices = [child["logit_index"] for child in node["children"]]
                    if len(child_indices) > 0:
                        softmax_probs[:, child_indices] = F.softmax(output[:, child_indices], dim=1)
        else:
            softmax_probs = F.softmax(output, dim=1)
            
        return softmax_probs
    
    def compute_ood_scores(self, output, preds, feats=None):
        output = self.normalise_probs(output)
        if self.use_hierarchy:
            batch_size = output.size(0)
            mean_entropies, min_entropies, max_entropies = [], [], []

            for i in range(batch_size):
                path = self.paths[preds[i].item()]
                path_entropies = []
                for node in path:
                    if node["children"]:
                        node_id = node["logit_index"]

                        children = node["children"]
                        child_indices = [child["logit_index"] for child in children]
                        child_probs = torch.tensor([output[i, child_index] for child_index in child_indices])
                        child_entropy = -torch.sum(child_probs * torch.log(child_probs + 1e-8))
                        
                        if self.store_entropy:
                            self.entropy_tracker.node_entropies[node_id].append(child_entropy.item())
                        path_entropies.append(child_entropy.item())
                mean_entropies.append(torch.mean(torch.tensor(path_entropies)))
                min_entropies.append(torch.min(torch.tensor(path_entropies)))
                max_entropies.append(torch.max(torch.tensor(path_entropies)))

            return (
                torch.tensor(mean_entropies, device=output.device), 
                torch.tensor(min_entropies, device=output.device), 
                torch.tensor(max_entropies, device=output.device)
            )
        else:
            return (
                torch.zeros(output.size(0),device=output.device), 
                torch.zeros(output.size(0), device=output.device), 
                torch.zeros(output.size(0), device=output.device)
            )
    

    def transform_linkage_matrix(self):
        n = self.Z.shape[0] + 1  # Number of initial clusters
        result = []
        cluster_nodes = {}  # Dictionary to store all nodes (original and internal) for each cluster

        # Initialize nodes for leaf clusters (original data points)
        for i in range(n):
            cluster_nodes[i] = [i]

        # Iterate through each row of the linkage matrix
        for i in range(self.Z.shape[0]):
            cluster_index = n + i  # Index of the new internal cluster node
            ward_distance = self.Z[i, 2]  # Ward distance when the cluster was created

            # Get the two clusters being merged
            cluster1 = int(self.Z[i, 0])
            cluster2 = int(self.Z[i, 1])

            # Get the nodes of the two clusters
            nodes1 = cluster_nodes.get(cluster1, [cluster1])
            nodes2 = cluster_nodes.get(cluster2, [cluster2])

            # Combine the nodes to form the new cluster's nodes
            # Include the new cluster index itself, as well as the nodes of the merged clusters
            combined_nodes = [cluster_index] + nodes1 + nodes2

            # Store the nodes for the new cluster
            cluster_nodes[cluster_index] = combined_nodes

            # Append the result as a new row
            result.append([cluster_index, ward_distance, combined_nodes])

        return np.array(result, dtype=object)
    
    
    def inference_with_stopping(self, output, preds):
        """
        Perform inference with early stopping based on entropy thresholds.
        Returns predictions tensor where each element is either:
        - logit_index of the final predicted node (if reached a leaf)
        - logit_index of the internal node where we stopped (if stopped early due to high entropy)
        """
        output = self.normalise_probs(output)
        batch_size = output.size(0)
        predictions = torch.zeros(batch_size, dtype=torch.long, device=output.device)
        
        for i in range(batch_size):
            path = self.paths[preds[i].item()]
            current_node = path[0]

            for node in path:
                if node["children"]:
                    children = node["children"]
                    child_indices = [child["logit_index"] for child in children]
                    child_probs = torch.tensor([output[i, child_index] for child_index in child_indices])
                    child_entropy = -torch.sum(child_probs * torch.log(child_probs))
                    if self.entropy_tracker.should_stop(node["logit_index"], child_entropy.item()):
                        predictions[i] = node["logit_index"]
                        break                
                elif not node["children"]:
                    predictions[i] = node["logit_index"]
                    break
        return predictions
    




class HierarchicalEntropyTracker:
    def __init__(self):
        self.node_entropies: Dict[str, List[float]] = defaultdict(list)
        self.node_thresholds: Dict[str, float] = {}
    
    def clear(self):
        """Clear accumulated entropy values to prevent memory leaks."""
        self.node_entropies.clear()
        self.node_thresholds.clear()
    
    def compute_thresholds(self, lambda_hat: float = 0.99):
        """Compute node-specific thresholds from collected entropy values."""
        for node_id, entropies in self.node_entropies.items():
            if entropies:
                self.node_thresholds[node_id] = torch.tensor(entropies).quantile(lambda_hat).item()
    
    def should_stop(self, node_id: str, entropy: float) -> bool:
        """Check if inference should stop at current node based on entropy threshold."""
        if node_id not in self.node_thresholds:
            return False
        return entropy > self.node_thresholds[node_id]
