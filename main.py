import torch
import hydra
import pandas as pd
import wandb
import logging
import os
import numpy as np

from omegaconf import OmegaConf, ListConfig
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from torcheval.metrics.functional import multiclass_f1_score
from torchmetrics import ConfusionMatrix

from utils import initialise_optimizer, initialise_model,  load_from_npy, get_labels, setup_data, downsample_X, EarlyStopping, separate_X_sensorwise, get_train_test_idxs
from hierarchy_generation.dl_generation import initialise_hierarchy

def train_test_model(cfg, X_feats, y, groups, my_device):
    y, le, labels = get_labels(y)
    if isinstance(X_feats, pd.DataFrame):
        X_feats = X_feats.to_numpy()

    # K-fold cross-validation through the subjects
    # Specified in cfg.evaluation.set_folds=False
    # If != False, uses the specified folds in form of a list of lists of subject IDs
    if cfg.evaluation.set_folds:        
        if isinstance(cfg.evaluation.set_folds, ListConfig):
            folds = []
            for fold in cfg.evaluation.set_folds:
                if isinstance(fold, ListConfig):
                    folds.append(list(fold))
                else:
                    folds.append(fold)
        else:
            folds = (cfg.evaluation.set_folds)
    # K-fold based on number of subjects
    else:  
        all_subjects = sorted(list(set(groups)))
        num_subjects = len(all_subjects)
        if num_subjects < 2:
            raise ValueError("Not enough subjects for cross-validation. At least 2 subjects are required.")
        elif num_subjects < 5:
            # For small number of subjects, place each subject in its own fold
            folds = [[subject] for subject in all_subjects]
        else:
            # For 5 or more subjects, create folds with 2 subjects in each fold
            # This creates a list where each inner list contains 2 subjects
            folds = []
            for i in range(0, num_subjects, 2):
                if i + 1 < num_subjects:
                    folds.append([all_subjects[i], all_subjects[i+1]])
                else:
                    # If odd number of subjects, last fold has only one subject
                    folds.append([all_subjects[i]])


    # Tables to save test results
    id_columns = ["run_id","fold","f1", "acc"]
    id_table = wandb.Table(columns=id_columns)
    ood_columns = ["run_id", "fold", "auroc", "fpr95", "detection_error"]
    ood_table = wandb.Table(columns=ood_columns)
    id_f1_total, auroc_total = 0, 0

    # Begin training/testing
    for fold in folds:
        print("=" * 60)
        print("Running fold with subjects:", fold)

        # Select specific subjects/activities for testing
        train_idx, test_id_idx, test_ood_idx = get_train_test_idxs(cfg, y, groups, fold)

        # Set up hierarchy, extracting deep learning features
        print("Setting up hierarchy...")
        initialise_hierarchy(cfg, my_device, X_feats, y, groups, fold)
        print("Hierarchy setup complete.\n")

        # Get loaders up
        train_loader, val_loader, test_id_loader, test_ood_loader, weights = setup_data(
            train_idx, test_id_idx, test_ood_idx, X_feats, y, groups, cfg, class_weights=cfg.model.weighted, my_device=my_device
        ) 
        
        torch.cuda.empty_cache() 
        early_stopping = EarlyStopping(
            patience=cfg.evaluation.patience, path=cfg.model_path, verbose=True
        )
        torch.cuda.empty_cache()

        # Initialize model
        model = initialise_model(cfg, my_device, weights)
        torch.cuda.empty_cache()

        # Get optimiser
        optimizer = initialise_optimizer(cfg, model)
        torch.cuda.empty_cache()
        
        print("Training...")
        _, _ = train_model(cfg,
                           model,
                           optimizer,
                           train_loader,
                           val_loader,
                           early_stopping,
                           fold
                           )

        print("Testing...") 
        model = initialise_model(cfg, my_device, weights)
        model = model.to(my_device, dtype=torch.float)
        model.load_state_dict(torch.load(cfg.model_path))

        id_data, ood_data, id_f1, auroc_mean = test_model(cfg, 
                                                        model, 
                                                        test_id_loader, 
                                                        test_ood_loader,
                                                        fold
                                                        )
        id_table.add_data(*[id_data[col] for col in id_columns])
        ood_table.add_data(*[ood_data[col] for col in ood_columns])
        id_f1_total += id_f1
        auroc_total += auroc_mean
        
        # Clean up model and GPU memory after each fold
        del model, optimizer, train_loader, val_loader, test_id_loader, test_ood_loader
        torch.cuda.empty_cache()
        print(f"Completed fold {fold}")
    
    wandb.log({f"test/ID_table":id_table})
    wandb.log({f"test/OOD_table":ood_table})
    id_f1_total /= len(folds)
    auroc_total /= len(folds)
    print(f"Average ID F1 score: {id_f1_total:.4f}")
    print(f"Average OOD AUROC: {auroc_total:.4f}")
    wandb.log({
        "test/average_id_f1": id_f1_total,
        "test/average_auroc": auroc_total,
    })
    # Delete hierarchy YAML file generated this run
    hier_output_path = f"{cfg.tree.HAC_root}/{wandb.run.id}"
    if os.path.isfile(f"{hier_output_path}.yaml"):
        try:
            # Only delete if not saving hierarchy
            os.remove(f"{hier_output_path}.yaml")
            print(f"Successfully deleted hierarchy YAML file at {hier_output_path}.yaml")
            os.remove(f"{hier_output_path}_Z.npy")
            print(f"Successfully deleted Z file at {hier_output_path}_Z.npy")
        except Exception as e:
            print(f"Failed to delete hierarchy YAML file: {e}")
    else:
        print(f"Hierarchy YAML file not found at {hier_output_path}.yaml")


def train_model(cfg, model, optimizer, train_loader, val_loader, early_stopping, fold):
    torch.cuda.empty_cache()
    
    for epoch in range(cfg.evaluation.num_epoch):
        torch.cuda.empty_cache()
        
        model.train()
        train_losses = 0

        for i, (my_X, my_Y) in enumerate(train_loader):            
            torch.cuda.empty_cache()
            my_X_separated = separate_X_sensorwise(cfg, my_X, model.module.device)
            del my_X
            
            true_y = my_Y.to(model.module.device, dtype=torch.long)
            del my_Y  

            # Calculate loss 
            logits, feats = model(my_X_separated)
            loss, loss_other, loss_soft = model.module.compute_loss(logits, true_y, feats)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Clear gradients immediately after step
            for param in model.parameters():
                param.grad = None

            train_losses += loss.item()
            
            # Log scalars only to avoid accumulating tensors
            wandb.log({
                f"train/loss_{fold}": loss.item(),
                f"train/epoch_{fold}": epoch,
                f"train/loss_other_{fold}": loss_other.item() if isinstance(loss_other, torch.Tensor) else loss_other,
                f"train/loss_soft_{fold}": loss_soft.item()
            })
            
            del my_X_separated, true_y, logits, feats, loss, loss_other, loss_soft
            torch.cuda.empty_cache()
            
        train_losses /= len(train_loader)
        val_loss, val_other, val_soft, val_acc, val_f1 = evaluate_model(model, val_loader, cfg)

        # print + log validation results
        print_msg = (
            f"[{epoch}/{cfg.evaluation.num_epoch}] "
            + f"train_loss: {train_losses:.4e} "
            + f"val_loss: {val_loss:.4e} "
            + f"val_f1: {val_f1:.4f} " 
            + f"val_acc: {val_acc:.4f} " 
            + f"val_other: {val_other:.4e} " 
            + f"val_soft: {val_soft:.4e} "
            )
        early_stopping(val_loss, model)
        print(print_msg)
        wandb.log({
                f"train/loss_epochwise_{fold}":train_losses, f"validation/loss_{fold}":val_loss, 
                f"validation/f1_{fold}":val_f1, f"validation/acc_{fold}":val_acc,
                f"validation/loss_other_{fold}":val_other, f"validation/loss_soft_{fold}":val_soft,
        })
        torch.cuda.empty_cache()
        
        # Clear entropy tracker
        if hasattr(model.module, 'entropy_tracker'):
            model.module.entropy_tracker.clear()

        # Early stopping
        if early_stopping.early_stop:
            print("Early stopping")
            break
            
    return True, True

def evaluate_model(model, val_loader, cfg):
    model.eval()
    val_loss, val_other, val_soft, all_preds, all_ys = 0.0, 0.0, 0.0, [], []

    with torch.no_grad():
        for i, (my_X, my_Y) in enumerate(val_loader):
            my_X_separated = separate_X_sensorwise(cfg, my_X, model.module.device)
            del my_X  # Delete original tensor
            
            true_y = my_Y.to(model.module.device, dtype=torch.long)
            del my_Y  # Delete original tensor

            logits, feats = model(my_X_separated)
            loss, loss_other, loss_soft = model.module.compute_loss(logits, true_y, feats)

            val_loss += loss.item()
            if isinstance(loss_other, torch.Tensor):
                val_other += loss_other.item()
            else:
                val_other += loss_other
            val_soft += loss_soft.item()

            preds = model.module.predict(logits)
            all_preds.extend(preds.cpu().numpy())
            all_ys.extend(true_y.cpu().numpy())
            
            # Clean up tensors to prevent memory accumulation
            del my_X_separated, true_y, logits, feats, loss, loss_other, loss_soft, preds

    val_loss /= len(val_loader)
    val_other /= len(val_loader)
    val_soft /= len(val_loader)
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)
    val_f1 = multiclass_f1_score(torch.tensor(all_preds), torch.tensor(all_ys), num_classes=len(cfg.data.label2code)+1, average="macro")
    val_acc = accuracy_score(all_ys, all_preds)

    return val_loss, val_other, val_soft, val_acc, val_f1

def test_model(cfg, model, id_loader, ood_loader, fold):
    """
    Test model on both ID and OOD metrics
    test set is separated into ID and OOD samples
    1. ID metrics: accuracy, f1
    2. OOD metrics: AUROC, FPR95, Detection Error
    """
    model.eval()
    test_loss, id_preds, id_ys = 0.0, [], []
    ood_scores, ood_preds, ood_ys = [], [], []
    mean_entropy, min_entropy, max_entropy = [], [], []
    outputs_id, outputs_ood = torch.zeros(0, model.module.num_output_nodes, device=model.module.device), torch.zeros(0, model.module.num_output_nodes, device=model.module.device)

    # ID
    with torch.no_grad():
        for i, (my_X, my_Y, pid) in enumerate(id_loader):
            my_X_separated = separate_X_sensorwise(cfg, my_X, model.module.device)
            del my_X  
            
            true_y = my_Y.to(model.module.device, dtype=torch.long)
            del my_Y  

            logits, feats = model(my_X_separated)
            loss, loss_other, loss_soft = model.module.compute_loss(logits, true_y, feats)
            test_loss += loss.item()

            preds = model.module.predict(logits)
            # Compute OOD scores (path entropy)
            mean_e, min_e, max_e = model.module.compute_ood_scores(logits, preds, feats)
            
            outputs_id = torch.cat((outputs_id, logits), 0)
            id_preds.extend(preds.cpu().numpy())
            id_ys.extend(true_y.cpu().numpy())
            mean_entropy.extend(mean_e.cpu().numpy())
            min_entropy.extend(min_e.cpu().numpy())
            max_entropy.extend(max_e.cpu().numpy())
            
            del my_X_separated, true_y, logits, feats, loss, loss_other, loss_soft, preds, mean_e, min_e, max_e
            torch.cuda.empty_cache()
            
    # OOD 
    with torch.no_grad():
        for i, (my_X, my_Y, pid) in enumerate(ood_loader):
            my_X_separated = separate_X_sensorwise(cfg, my_X, model.module.device)
            del my_X  # Delete original tensor
            
            true_y = my_Y.to(model.module.device, dtype=torch.long)
            del my_Y  # Delete original tensor

            logits, feats = model(my_X_separated)

            preds = model.module.predict(logits)
            # Compute OOD scores
            mean_e, min_e, max_e = model.module.compute_ood_scores(logits, preds, feats)

            outputs_ood = torch.cat((outputs_ood, logits), 0)
            ood_preds.extend(preds.cpu().numpy())
            ood_ys.extend(true_y.cpu().numpy())
            mean_entropy.extend(mean_e.cpu().numpy())
            min_entropy.extend(min_e.cpu().numpy())
            max_entropy.extend(max_e.cpu().numpy())
            
            # Clean up tensors immediately  
            del my_X_separated, true_y, logits, feats, preds, mean_e, min_e, max_e
            torch.cuda.empty_cache()

    test_loss /= len(id_loader)

    # ID metrics (accuracy, f1)
    id_acc = accuracy_score(id_ys, id_preds)
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)
    id_f1 = multiclass_f1_score(torch.tensor(id_preds), torch.tensor(id_ys), num_classes=len(cfg.data.label2code)+1, average="macro")
    if 'save_cmat' in cfg.model and cfg.model.save_cmat:
        confmat = ConfusionMatrix(task='multiclass', num_classes=len(cfg.data.label2code))
        mycmat  = confmat(torch.tensor(id_preds), torch.tensor(id_ys))
        mycmat = mycmat.cpu().numpy()
        np.save(f"{wandb.run.dir}/confusion_matrix_{fold}.npy", mycmat)
    
    # OOD metrics
    # AUC, FPR95, Detection Error
    ood_labels = [0]*len(id_preds)+[1]*len(ood_preds)
    if len(ood_loader) > 0:
        auroc_mean, fpr95_mean, detection_error_mean = ood_calculations(ood_labels, mean_entropy)
        
    # In case of no excluded classes (no OOD)
    else:
        auroc_mean, fpr95_mean, detection_error_mean = 0,0,0
    
    # Print + log results
    print_msg = (
        f"Test Results: "
        + f"test_loss: {test_loss:.4e} "
        + f"id_acc: {id_acc:.4f} " 
        + f"id_f1: {id_f1:.4f} " 
        + f"auroc_mean: {auroc_mean:.4f} "
        )
    print(print_msg)
    wandb.log({
            f"test/loss_{fold}":test_loss
    })
    id_data = {"run_id":wandb.run.id, "fold":fold, "f1":id_f1, "acc":id_acc}
    ood_data = {"run_id":wandb.run.id, "fold":fold, "auroc":auroc_mean, "fpr95":fpr95_mean, "detection_error":detection_error_mean}

    return id_data, ood_data, id_f1, auroc_mean

def ood_calculations(ood_labels, ood_scores):
    auroc = roc_auc_score(ood_labels, ood_scores)
    fpr, tpr, thresholds = roc_curve(ood_labels, ood_scores)
    target_index = np.argmin(np.abs(tpr - 0.95))
    fpr95 = fpr[target_index]
    fnr = 1 - tpr  
    detection_errors = 0.5 * (fpr + fnr)
    min_detection_error = np.min(detection_errors)

    return auroc, fpr95, min_detection_error


@hydra.main(config_path="../Hi_OSCAR/conf", config_name="main", version_base=None)
def main(cfg): 
    #################
    # WandB logging:
    wandb.init(
        project=cfg.wandb.project,
        job_type=cfg.wandb.job_type,
        notes=cfg.wandb.notes,
        config=OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        settings=wandb.Settings(start_method="thread")
    )

    #################
    # Prepare data + run training/testing
    my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, Y, P = load_from_npy(cfg)
    print("Original X shape:", X.shape)

    # Adjust input size based on window size
    window_size = int(cfg.evaluation.window_size)
    assert window_size in [5, 10], "Evaluation window size must be 5 or 10 seconds."
    input_size = int(cfg.evaluation.window_size) * 30 # 300 for w10, 150 for w05
    X_downsampled = downsample_X(X, input_size)

    train_test_model(cfg, X_downsampled, Y, P, my_device)

if __name__ == "__main__":
    main()