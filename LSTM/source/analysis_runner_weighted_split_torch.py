import os
import time
import math
import sys

sys.path.append("..")
sys.path.append("../..")

from source.analysis.model import LocalGlobalLSTM
from source.analysis.setup.attributed_classifier import AttributedClassifier
from source.analysis.setup.subject_builder import SubjectBuilder
from source.analysis.setup.train_test_splitter import TrainTestSplitter
from source.analysis.classification.classifier_service import ClassifierService
from source.analysis.classification.classifier_summary import ClassifierSummary
from source.analysis.performance.raw_performance import RawPerformance

from source import utils
from source.constants import Constants
from preprocessing.psg.psg_label_service import PSGLabelService

import torch
import random
import numpy as np
import pandas as pd


def set_random_seed(seed=None):
    if seed is not None and not (isinstance(seed, int) and 0 <= seed):
        raise ValueError('Seed must be a non-negative integer or omitted, not {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def get_device():
    """Detect best available device: MPS (Apple Silicon), CUDA, or CPU."""
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_fold_from_disk(subject_ids, pred_dir, model_dir):
    """
    Reconstruct a RawPerformance from already-saved fold outputs.
    Called when a fold's prediction CSV is detected on disk so training
    can be skipped on re-runs.

    Feature importance is set to None because the values are already
    captured in any existing feature_importance.csv from the prior run.
    Model state dict is set to None because the checkpoint is already on disk.
    """
    true_labels_list, pred_labels_list, class_probs_list = [], [], []
    for sid in subject_ids:
        df = pd.read_csv(os.path.join(pred_dir, f'ML_pred_{sid}_lstm.csv'))
        true_labels_list.append(df['label_n'].values)
        pred_labels_list.append(df['.pred_class'].values)
        class_probs_list.append(
            np.stack([df['.pred_0'].values, df['.pred_1'].values], axis=1)
        )
    return RawPerformance(
        true_labels=true_labels_list,
        class_probabilities=class_probs_list,
        subject=list(subject_ids),
        predicted_labels=pred_labels_list,
        feats=None,           # already in feature_importance.csv from prior run
        model_state_dict=None,  # checkpoint already on disk
    )


def save_predictions(classifier_summary: ClassifierSummary, output_dir: str):
    """
    Save per-epoch predictions as one CSV per subject, matching the format of
    ML_pred_1_lstm.csv:

        timestamp   — Unix epoch seconds for the 30-s window
        .pred_1     — softmax probability of class 1 (Sleep)
        .pred_0     — softmax probability of class 0 (Wake)
        .pred_class — predicted class  (0=Wake, 1=Sleep)
        label_n     — ground-truth label

    Files are written to:
        <output_dir>/<classifier_name>/<feature_label>/predictions/
            ML_pred_<subject_id>_lstm.csv

    Model checkpoints (one per LOSO fold) are written to:
        <output_dir>/<classifier_name>/<feature_label>/models/
            fold_<subject_id>.pt

    Feature importance (one row per feature per fold) is written to:
        <output_dir>/<classifier_name>/<feature_label>/
            feature_importance.csv
    """
    for feature_set in classifier_summary.performance_dictionary:
        feature_label = '+'.join([i.value for i in feature_set])
        base_path = os.path.join(output_dir,
                                 classifier_summary.attributed_classifier.name,
                                 feature_label)

        pred_dir   = os.path.join(base_path, 'predictions')
        model_dir  = os.path.join(base_path, 'models')
        os.makedirs(pred_dir,  exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        raw_performances = classifier_summary.performance_dictionary[feature_set]

        importance_rows = []   # accumulated across all folds for summary CSV
        total_epochs = 0

        for performance in raw_performances:
            # In LOSO each RawPerformance holds exactly one test subject.
            # Guard: iterate in case a fold somehow has multiple subjects.
            n_subjects = len(performance.true_labels)
            for i in range(n_subjects):
                subject_id = performance.subject[i] if performance.subject is not None else f'subject_{i}'

                true_labels   = np.array(performance.true_labels[i])
                pred_labels   = np.array(performance.predicted_labels[i])
                class_probs   = np.array(performance.class_probabilities[i])  # shape (N, n_class)

                # Probabilities for each class
                prob_sleep = class_probs[:, 1]   # .pred_1 — P(Sleep)
                prob_wake  = class_probs[:, 0]   # .pred_0 — P(Wake)

                # Load epoch-level timestamps
                try:
                    timestamps = PSGLabelService.load_timestamps(subject_id)
                    n = min(len(timestamps), len(true_labels))
                    timestamps  = timestamps[:n]
                    true_labels = true_labels[:n]
                    pred_labels = pred_labels[:n]
                    prob_sleep  = prob_sleep[:n]
                    prob_wake   = prob_wake[:n]
                except Exception as exc:
                    print(f'  Warning: could not load timestamps for {subject_id}: {exc}')
                    timestamps = np.arange(len(true_labels), dtype=float)

                # ── Per-subject prediction CSV ─────────────────────────────────
                subject_df = pd.DataFrame({
                    'timestamp':   timestamps,
                    '.pred_1':     np.round(prob_sleep, 6),
                    '.pred_0':     np.round(prob_wake, 6),
                    '.pred_class': pred_labels.astype(int),
                    'label_n':     true_labels.astype(int),
                })
                out_csv = os.path.join(pred_dir, f'ML_pred_{subject_id}_lstm.csv')
                if os.path.exists(out_csv):
                    print(f'  [{subject_id}]  skipping CSV (already exists)')
                else:
                    subject_df.to_csv(out_csv, index=False)
                    print(f'  [{subject_id}]  {len(subject_df):,} epochs → {out_csv}')
                total_epochs += len(subject_df)

                # ── Model checkpoint ──────────────────────────────────────────
                if performance.model_state_dict is not None:
                    ckpt_path = os.path.join(model_dir, f'fold_{subject_id}.pt')
                    if os.path.exists(ckpt_path):
                        if Constants.VERBOSE:
                            print(f'  Checkpoint already exists — skipping → {ckpt_path}')
                    else:
                        torch.save(performance.model_state_dict, ckpt_path)
                        if Constants.VERBOSE:
                            print(f'  Checkpoint saved → {ckpt_path}')

                # ── Collect feature importance for this fold ──────────────────
                if performance.feature_importance:
                    for feat_name, importance in performance.feature_importance.items():
                        importance_rows.append({
                            'subject':    subject_id,
                            'feature':    feat_name,
                            'importance': importance,
                        })

        print(f'\n  Total epochs written: {total_epochs:,}')
        print(f'  Prediction CSVs    → {pred_dir}/')
        print(f'  Model checkpoints  → {model_dir}/')

        # ── Feature importance summary CSV ────────────────────────────────────
        imp_csv = os.path.join(base_path, 'feature_importance.csv')
        if importance_rows:
            new_imp_df = pd.DataFrame(importance_rows)

            # Merge with any existing importance rows from a prior (partial) run.
            # Rows from the current run take precedence for the same subject.
            if os.path.exists(imp_csv):
                existing = pd.read_csv(imp_csv)
                # Drop the old __mean__ summary row and any subjects re-computed now
                existing = existing[existing['subject'] != '__mean__']
                new_subjects = new_imp_df['subject'].unique()
                existing = existing[~existing['subject'].isin(new_subjects)]
                new_imp_df = pd.concat([existing, new_imp_df], ignore_index=True)

            # Recompute mean across all folds (existing + new)
            mean_imp = (new_imp_df.groupby('feature')['importance']
                        .mean().reset_index()
                        .rename(columns={'importance': 'importance'})
                        .assign(subject='__mean__'))
            imp_df = pd.concat([new_imp_df, mean_imp[['subject', 'feature', 'importance']]],
                               ignore_index=True)
            imp_df.to_csv(imp_csv, index=False)
            print(f'  Feature importance → {imp_csv}')
        elif os.path.exists(imp_csv):
            print(f'  Feature importance → {imp_csv} (no new folds; existing file unchanged)')

        # ── Best hyperparameters per fold CSV ────────────────────────────────
        hp_rows = []
        for performance in raw_performances:
            if performance.best_hyperparams:
                n_subjects = len(performance.true_labels)
                for i in range(n_subjects):
                    subject_id = performance.subject[i] if performance.subject is not None else f'subject_{i}'
                    row = {'subject': subject_id}
                    row.update(performance.best_hyperparams)
                    hp_rows.append(row)

        hp_csv = os.path.join(base_path, 'best_hyperparameters.csv')
        if hp_rows:
            new_hp_df = pd.DataFrame(hp_rows)

            # Merge with existing rows from a prior partial run
            if os.path.exists(hp_csv):
                existing = pd.read_csv(hp_csv)
                new_subjects = new_hp_df['subject'].unique()
                existing = existing[~existing['subject'].isin(new_subjects)]
                new_hp_df = pd.concat([existing, new_hp_df], ignore_index=True)

            new_hp_df.to_csv(hp_csv, index=False)
            print(f'  Best hyperparameters → {hp_csv}')
        elif os.path.exists(hp_csv):
            print(f'  Best hyperparameters → {hp_csv} (no new folds; existing file unchanged)')


def train(feature_sets, output_dir, class_num=2, loso=True):
    """
    Train the LocalGlobalLSTM with fold-level skip logic.

    If a fold's prediction CSV already exists on disk (from a prior interrupted
    run), that fold is loaded from disk rather than re-trained.  This allows a
    SLURM job that was killed mid-run to resume from where it left off.

    Parameters
    ----------
    feature_sets : list
        Output of utils.get_lstm_feature_fft30_sets().
    output_dir : str
        Directory where outputs will be written.
    class_num : int
        2 = binary Wake/Sleep  (default for AGV pipeline).
    loso : bool
        True  → leave-one-subject-out CV  (recommended; default).
        False → 10-fold CV.
    """
    print(f'class_num={class_num}  loso={loso}')

    # Resolve output paths early so we can check for existing fold outputs
    # before starting any training.
    feature_label = '+'.join([i.value for i in feature_sets[0]])
    base_path  = os.path.join(output_dir, 'LocalGlobalLSTM', feature_label)
    pred_dir   = os.path.join(base_path, 'predictions')
    model_dir  = os.path.join(base_path, 'models')
    os.makedirs(pred_dir,  exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # NOTE: The model is now created inside the inner CV loop (classifier_service)
    # because hyperparameters (lr, dropout) are tuned per fold via
    # 5-fold grid search. A placeholder attributed_classifier is created per fold
    # with a dummy model; the real model is built inside the grid search.
    classifier_name = 'LocalGlobalLSTM'
    print(f'Running {classifier_name} with nested CV (LOSO outer, 5-fold grid search inner)...')

    subject_dictionary = SubjectBuilder.get_subject_dictionary()
    subject_ids = list(subject_dictionary.keys())

    if loso:
        data_splits = TrainTestSplitter.leave_one_out(subject_ids)
        print(f'LOSO CV across {len(subject_ids)} subjects ({len(data_splits)} folds)')
    else:
        test_index = max(1, math.ceil(len(subject_ids) / 10))
        data_splits = TrainTestSplitter.by_number(subject_ids, test_index)
        print(f'10-fold CV across {len(subject_ids)} subjects ({len(data_splits)} folds)')

    # Placeholder attributed_classifier — the actual model is built inside
    # classifier_service's 5-fold grid search.  Defined once before the loop
    # so it is always in scope for ClassifierSummary after the loop.
    attributed_classifier = AttributedClassifier(
        name=classifier_name,
        classifier=None,
    )

    raw_performances = []
    n_skipped = 0
    n_trained = 0

    for fold_idx, data_split in enumerate(data_splits):
        test_subjects = data_split.testing_set  # list of subject IDs for this fold

        # ── Skip check ────────────────────────────────────────────────────────
        # A fold is considered complete if every test subject already has a
        # prediction CSV on disk (written by a prior run of save_predictions).
        existing_csvs = [
            os.path.join(pred_dir, f'ML_pred_{sid}_lstm.csv')
            for sid in test_subjects
        ]
        all_done = all(os.path.exists(p) for p in existing_csvs)

        if all_done:
            print(f'  Fold {fold_idx + 1}/{len(data_splits)} '
                  f'[{", ".join(test_subjects)}] — skipping (already on disk)')
            perf = load_fold_from_disk(test_subjects, pred_dir, model_dir)
            raw_performances.append(perf)
            n_skipped += 1
            continue

        # ── Run this fold ─────────────────────────────────────────────────────
        print(f'  Fold {fold_idx + 1}/{len(data_splits)} '
              f'[{", ".join(test_subjects)}] — training...')
        perf = ClassifierService.run_single_data_split_sw(
            data_split, attributed_classifier, subject_dictionary, feature_sets[0]
        )
        raw_performances.append(perf)
        n_trained += 1

    print(f'\nFolds complete: {n_trained} trained, {n_skipped} skipped (loaded from disk)')

    performance_dictionary = {tuple(feature_sets[0]): raw_performances}
    classifier_summary = ClassifierSummary(attributed_classifier, performance_dictionary)
    save_predictions(classifier_summary, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train AGV LSTM sleep-wake classifier.')
    parser.add_argument('--output-dir', default=None,
                        help='Directory for outputs. '
                             'Defaults to ../outputs/agv_loso/ relative to this script.')
    parser.add_argument('--no-loso', action='store_true',
                        help='Use 10-fold CV instead of leave-one-subject-out.')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_random_seed(args.seed)
    print(f'Device: {Constants.DEVICE}  |  PyTorch device: {get_device()}')

    feature_sets = utils.get_lstm_feature_fft30_sets()

    if args.output_dir is None:
        output_dir = os.path.join('..', 'outputs', f'{Constants.DEVICE}_loso')
    else:
        output_dir = args.output_dir

    start_time = time.time()
    train(feature_sets,
          output_dir=output_dir,
          class_num=2,
          loso=not args.no_loso)
    elapsed = (time.time() - start_time) / 60
    print(f'Elapsed: {elapsed:.1f} minutes')
