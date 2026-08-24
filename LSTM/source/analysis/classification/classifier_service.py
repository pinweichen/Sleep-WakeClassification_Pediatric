import time
import copy
import itertools

import numpy as np
from sklearn.utils import class_weight
from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader

from source.analysis.classification.classifier_input_builder import ClassifierInputBuilder
from source.analysis.dataset import TimeSeriesDataset, collate_fn
from source.analysis.performance.raw_performance import RawPerformance
from source.constants import Constants

import torch

from source.analysis.model import LocalGlobalLSTM, Trainer


# Feature names match the order in utils.get_lstm_feature_fft30_sets()
_FEATURE_NAMES = ['motion_xfft1_30', 'motion_yfft1_30', 'motion_zfft1_30', 'motion_vmfft1_30']

# ── Hyperparameter grid ──────────────────────────────────────────────────────
PARAM_GRID = {
    'lr': [0.001, 0.0001, 0.00001],
    'dropout': [0.1, 0.3, 0.5],
}


def _get_torch_device():
    """Return the best available device: CUDA → MPS (Apple Silicon) → CPU."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def _build_param_combinations(param_grid):
    """Expand a dict of lists into a list of dicts (full grid)."""
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


class ClassifierService(object):

    @staticmethod
    def run_sw(data_splits, classifier, subject_dictionary, feature_set):
        return ClassifierService.run_in_parallel(ClassifierService.run_single_data_split_sw,
                                                 data_splits, classifier,
                                                 subject_dictionary, feature_set)

    @staticmethod
    def run_four_class(data_splits, classifier, subject_dictionary, feature_set):
        return ClassifierService.run_in_parallel(ClassifierService.run_single_data_split_four_class,
                                                 data_splits, classifier,
                                                 subject_dictionary, feature_set)

    @staticmethod
    def run_in_parallel(function, data_splits, classifier, subject_dictionary, feature_set):
        results = [function(i, classifier, subject_dictionary, feature_set) for i in data_splits]
        return results

    @staticmethod
    def run_single_data_split_sw(data_split, attributed_classifier, subject_dictionary, feature_set):

        training_x, training_y = ClassifierInputBuilder.get_sleep_wake_inputs(data_split.training_set,
                                                                              subject_dictionary=subject_dictionary,
                                                                              feature_set=feature_set)
        testing_x, testing_y = ClassifierInputBuilder.get_sleep_wake_inputs(data_split.testing_set,
                                                                            subject_dictionary=subject_dictionary,
                                                                            feature_set=feature_set)

        return ClassifierService.run_single_data_split(training_x, training_y, testing_x, testing_y,
                                                       attributed_classifier, testing_ids=data_split.testing_set)

    @staticmethod
    def run_single_data_split_four_class(data_split, attributed_classifier, subject_dictionary, feature_set):

        training_x, training_y = ClassifierInputBuilder.get_four_class_inputs(data_split.training_set,
                                                                              subject_dictionary=subject_dictionary,
                                                                              feature_set=feature_set)

        testing_x, testing_y = ClassifierInputBuilder.get_four_class_inputs(data_split.testing_set,
                                                                            subject_dictionary=subject_dictionary,
                                                                            feature_set=feature_set)

        return ClassifierService.run_single_data_split(training_x, training_y, testing_x, testing_y,
                                                       attributed_classifier, testing_ids=data_split.testing_set)

    @staticmethod
    def run_single_data_split(training_x, training_y, testing_x, testing_y, attributed_classifier, testing_ids=None):
        """
        Nested CV inner loop:
        1. KNN impute training and test features (no Z-score normalization —
           FFT spectral power values are used as-is).
        2. 5-fold CV grid search over (lr, dropout)
           to find the best hyperparameters maximizing AUC.
        3. Retrain on full training set with best hyperparameters.
        4. Evaluate on the held-out test subject.
        """
        start_time = time.time()

        # ── Step 1: KNN imputation only (no normalization) ───────────────────
        training_x, imputer = ClassifierInputBuilder.impute(training_x)

        # ── Step 1b: Transform test features using fitted imputer ────────────
        testing_x = ClassifierInputBuilder.transform_with_fitted_imputer(testing_x, imputer)

        # ── Compute class weights from training labels ────────────────────────
        all_labels = np.concatenate(training_y)
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(all_labels),
                                                          y=all_labels)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

        device = _get_torch_device()

        # ── Step 3: 5-fold CV grid search ─────────────────────────────────────
        dataset = TimeSeriesDataset(training_x, training_y)
        n_subjects = len(dataset)
        param_combos = _build_param_combinations(PARAM_GRID)

        print(f'    Inner CV: {len(param_combos)} hyperparameter combinations × 5 folds')

        best_mean_auc = -1.0
        best_params = param_combos[0]

        for p_idx, params in enumerate(param_combos):
            lr = params['lr']
            dropout = params['dropout']

            fold_aucs = []
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(range(n_subjects))):
                train_subset = Subset(dataset, train_idx.tolist())
                val_subset = Subset(dataset, val_idx.tolist())

                # Build a fresh model with these hyperparameters
                model = LocalGlobalLSTM(
                    feature_dim=training_x[0].shape[-1],
                    local_hidden_dim=128,
                    global_hidden_dim=128,
                    dropout=dropout,
                    n_class=2,
                )

                # Inner CV uses 100 epochs — enough for lr=0.00001 to show
                # meaningful learning while keeping compute manageable.
                # The final retrain (step 4) uses the full 300 epochs.
                trainer = Trainer(model, num_epochs=100, class_weight=class_weights_tensor,
                                  device=str(device), lr=lr)
                trainer.set_train_data(train_subset)
                trainer.set_val_data(val_subset)

                trainer.fit()

                # The best val AUC achieved during training is the fold score
                fold_aucs.append(trainer.val_score)

            mean_auc = float(np.mean(fold_aucs))
            if Constants.VERBOSE:
                print(f'    Params {p_idx + 1}/{len(param_combos)}: '
                      f'lr={lr}, dropout={dropout} '
                      f'→ mean AUC={mean_auc:.4f}')

            if mean_auc > best_mean_auc:
                best_mean_auc = mean_auc
                best_params = params

        print(f'    Best hyperparameters: lr={best_params["lr"]}, '
              f'dropout={best_params["dropout"]} '
              f'(mean AUC={best_mean_auc:.4f})')

        # ── Step 4: Retrain on full training set with best hyperparameters ────
        # Use an 80/20 split for early stopping during final retraining
        from torch.utils.data import random_split
        train_size = int(0.8 * n_subjects)
        val_size = n_subjects - train_size
        final_train_dataset, final_val_dataset = random_split(dataset, [train_size, val_size])

        final_model = LocalGlobalLSTM(
            feature_dim=training_x[0].shape[-1],
            local_hidden_dim=128,
            global_hidden_dim=128,
            dropout=best_params['dropout'],
            n_class=2,
        )

        test_dataset = TimeSeriesDataset(testing_x, testing_y)

        trainer = Trainer(final_model, class_weight=class_weights_tensor,
                          device=str(device), lr=best_params['lr'])
        trainer.set_train_data(final_train_dataset)
        trainer.set_val_data(final_val_dataset)
        trainer.set_test_data(test_dataset)
        trainer.fit()

        # ── Capture the best model state dict for this fold ───────────────────
        best_state = copy.deepcopy(trainer.best_model) if trainer.best_model is not None else None

        # ── Step 5: Evaluate on held-out test subject ─────────────────────────
        class_probabilities, predicted_labels, testing_y = trainer.test()

 
        # ── Permutation feature importance ────────────────────────────────────
        feature_importance = ClassifierService.compute_permutation_importance(
            model=final_model,
            test_dataset=test_dataset,
            device=str(device),
            feature_names=_FEATURE_NAMES,
            n_repeats=3,
        )

        raw_performance = RawPerformance(
            true_labels=testing_y,
            class_probabilities=class_probabilities,
            subject=testing_ids,
            predicted_labels=predicted_labels,
            feats=feature_importance,
            model_state_dict=best_state,
            best_hyperparams={
                'lr': best_params['lr'],
                'dropout': best_params['dropout'],
                'mean_cv_auc': best_mean_auc,
            },
        )

        if Constants.VERBOSE:
            print('Completed data split in ' + str(time.time() - start_time))

        return raw_performance

    # ── Permutation importance helpers ─────────────────────────────────────────

    @staticmethod
    def compute_permutation_importance(model, test_dataset, device, feature_names, n_repeats=3):
        """
        Estimate feature importance by permuting each feature type across epochs
        and measuring the resulting accuracy drop.

        For LOSO (1 test subject per fold), we permute the feature values
        across the epoch (time) dimension, which breaks the temporal signal
        carried by that channel while keeping its statistical distribution intact.

        Parameters
        ----------
        model        : trained LocalGlobalLSTM (weights loaded by Trainer.fit)
        test_dataset : TimeSeriesDataset for the held-out test subject(s)
        device       : torch device string
        feature_names: list of feature channel names  (length = feature_dim)
        n_repeats    : number of permutation repeats per feature for stable estimates

        Returns
        -------
        dict  {feature_name: mean_accuracy_drop}
        """
        loader = DataLoader(test_dataset, batch_size=5, shuffle=False, collate_fn=collate_fn)

        baseline_acc = ClassifierService._eval_accuracy(model, loader, device)

        importances = {}
        for f_idx, f_name in enumerate(feature_names):
            drops = []
            for _ in range(n_repeats):
                perm_acc = ClassifierService._eval_accuracy_permuted(model, loader, device, f_idx)
                drops.append(baseline_acc - perm_acc)
            importances[f_name] = float(np.mean(drops))
            if Constants.VERBOSE:
                print(f'  Permutation importance [{f_name}]: {importances[f_name]:.4f}')

        return importances

    @staticmethod
    def _eval_accuracy(model, loader, device):
        """Accuracy of the model on a DataLoader (no permutation)."""
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels, lengths in loader:
                inputs = inputs.to(device).float()
                outputs = model(inputs, lengths)
                outputs = outputs.reshape(-1, outputs.shape[-1])
                labels_flat = labels.reshape(-1)
                mask = labels_flat != -1
                preds = torch.argmax(outputs[mask], dim=-1).cpu().numpy()
                true  = labels_flat[mask].cpu().numpy()
                all_preds.append(preds)
                all_labels.append(true)
        all_preds  = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        return float((all_preds == all_labels).mean()) if len(all_labels) > 0 else 0.0

    @staticmethod
    def _eval_accuracy_permuted(model, loader, device, feature_idx):
        """
        Accuracy with one feature channel permuted across epochs.

        inputs shape: (batch, max_time, n_channels, n_features)
        We shuffle the time axis (dim=1) for the given feature_idx (dim=-1)
        within each sample, breaking the temporal signal of that channel.
        """
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels, lengths in loader:
                inputs = inputs.to(device).float()
                permuted = inputs.clone()
                batch_size = inputs.size(0)
                for b in range(batch_size):
                    T = lengths[b]   # actual epoch count for this subject (un-padded)
                    perm = torch.randperm(T)
                    permuted[b, :T, :, feature_idx] = inputs[b, perm, :, feature_idx]

                outputs = model(permuted, lengths)
                outputs = outputs.reshape(-1, outputs.shape[-1])
                labels_flat = labels.reshape(-1)
                mask = labels_flat != -1
                preds = torch.argmax(outputs[mask], dim=-1).cpu().numpy()
                true  = labels_flat[mask].cpu().numpy()
                all_preds.append(preds)
                all_labels.append(true)
        all_preds  = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        return float((all_preds == all_labels).mean()) if len(all_labels) > 0 else 0.0
