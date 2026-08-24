import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import trange
from sklearn.metrics import roc_auc_score
import copy
from source.analysis.dataset import collate_fn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LocalGlobalLSTM(nn.Module):
    def __init__(self, feature_dim=4, local_hidden_dim=128, global_hidden_dim=128, local_steps=15, dropout=0.1,
                 n_class=2):
        super(LocalGlobalLSTM, self).__init__()
        self.local_lstm = nn.LSTM(feature_dim, local_hidden_dim, batch_first=True, bidirectional=True)

        self.local_fc = nn.Sequential(
            nn.Linear(local_hidden_dim * local_steps * 2, local_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(local_hidden_dim, local_hidden_dim)
        )

        self.global_lstm = nn.LSTM(local_hidden_dim, global_hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(global_hidden_dim * 2, n_class)
        # NOTE: no Softmax here. forward() returns raw logits so that
        # nn.CrossEntropyLoss (used in Trainer) can apply log_softmax correctly.
        # CrossEntropyLoss internally computes log_softmax(logits) + NLLLoss.
        # If softmax were applied here, CrossEntropyLoss would compute
        # log_softmax(softmax(logits)), causing near-zero gradients and
        # locking the model at a trivial all-sleep prediction after ~5 epochs.

    def forward(self, x, lengths):
        batch_size, time_steps, num_channels, feature_dim = x.size()
        x = x.view(batch_size * time_steps, num_channels, feature_dim)
        local_lstm_out, _ = self.local_lstm(x)

        local_lstm_out = self.local_fc(local_lstm_out.reshape(batch_size * time_steps, -1))
        local_lstm_out = local_lstm_out.view(batch_size, time_steps, -1)

        packed_input = pack_padded_sequence(local_lstm_out, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.global_lstm(packed_input)
        global_lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        # Return raw logits — caller applies softmax if probabilities are needed.
        return self.fc(global_lstm_out)

class Trainer:
    def __init__(self, model, num_epochs=300, class_weight=None, device='cpu', lr=0.0001):
        self.batch_size = 5
        self.model = model
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.class_weight = class_weight.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.num_epochs = num_epochs
        self.val_score = 0
        self.best_model = None
        self.device = device
        self.collate_fn = collate_fn

    def set_train_data(self, dataset):
        self.train_set = dataset
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                                       collate_fn=self.collate_fn)

    def set_val_data(self, dataset):
        self.val_set = dataset
        self.val_loader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                                     collate_fn=self.collate_fn)

    def set_test_data(self, dataset):
        self.test_set = dataset
        self.test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False,
                                      collate_fn=self.collate_fn)

    def fit(self, early_stopping_patience=5):
        """
        Train the model for up to self.num_epochs epochs.

        Early stopping: if val_auc does not improve for `early_stopping_patience`
        consecutive epochs, training stops early and the best checkpoint is restored.
        patience=5 stops training after 5 consecutive epochs with no improvement.

        Fix: the original code had a `break` after the first batch of each epoch,
        meaning only 1 batch (5 subjects) was ever trained per epoch regardless of
        training-set size.  That line has been removed so every batch is used.
        """
        self.model = self.model.to(self.device)
        epochs_no_improve = 0

        for epoch in trange(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for inputs, labels, lengths in self.train_loader:
                inputs, labels = inputs.to(self.device).type(torch.float32), labels.to(self.device).long()
                self.optimizer.zero_grad()
                outputs = self.model(inputs, lengths)
                outputs = outputs.reshape(-1, outputs.shape[-1])
                labels = labels.reshape(-1)
                mask = (labels != -1)  
                outputs = outputs[mask]
                labels = labels[mask]
                loss = self.criterion(outputs, labels)
              
                weights = self.class_weight[labels]
                loss = loss * weights
                loss = loss.mean()

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(self.train_set)
            val_auc = self.eval()
            print(f'Epoch {epoch + 1}/{self.num_epochs} Loss: {epoch_loss:.4f}  Val AUC:{val_auc:.4f}')

            if val_auc > self.val_score:
                self.val_score = val_auc
                self.best_model = copy.deepcopy(self.model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= early_stopping_patience:
                    print(f'Early stopping at epoch {epoch + 1} '
                          f'(no val improvement for {early_stopping_patience} epochs)')
                    break

        print('Training complete')
        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)
        return self.model

    def eval(self):
        """
        Evaluate on the validation set and return **AUC** (Area Under the
        ROC Curve).

        Using AUC instead of raw accuracy as the early-stopping criterion
        prevents the model from being rewarded for predicting the majority
        class (sleep) exclusively.  AUC measures the model's ability to
        discriminate between classes across all decision thresholds, making
        it robust to class imbalance.  An all-sleep predictor scores an AUC
        of 0.50, so early stopping will not preserve a trivial model as the
        best checkpoint.
        """
        probabilities, gt = [], []
        self.model.eval()
        with torch.no_grad():
            for inputs, labels, lengths in self.val_loader:
                inputs = inputs.to(self.device).type(torch.float32)
                outputs = self.model(inputs, lengths)

                outputs = outputs.reshape(-1, outputs.shape[-1])
                labels = labels.reshape(-1)
                mask = (labels != -1)  
                outputs = outputs[mask]
                labels = labels[mask].cpu().numpy()

                probs = torch.softmax(outputs, dim=-1).cpu().numpy()
                splits = [probs[sum(lengths[:i]):sum(lengths[:i + 1])] for i in range(len(lengths))]
                splits_labels = [labels[sum(lengths[:i]):sum(lengths[:i + 1])] for i in range(len(lengths))]

                probabilities.append(splits)
                gt.append(splits_labels)

        probs_np = np.concatenate(sum(probabilities, []))
        gt_np = np.concatenate(sum(gt, []))

        # AUC requires both classes to be present in the ground truth
        if len(np.unique(gt_np)) < 2:
            return 0.5
        # Use probability of class 1 (wake) for binary AUC
        return float(roc_auc_score(gt_np, probs_np[:, 1]))

    def test(self):
        score, prediction, gt = [], [], []
        self.model.eval()
        with torch.no_grad():
            for inputs, labels, lengths in self.test_loader:
                inputs = inputs.to(self.device).type(torch.float32)
                outputs = self.model(inputs, lengths)  # raw logits

                outputs = outputs.reshape(-1, outputs.shape[-1])
                labels = labels.reshape(-1)
                mask = (labels != -1)  
                outputs = outputs[mask]
                labels = labels[mask].cpu().numpy()

                # Convert logits to probabilities for output CSVs and scoring.
                probs = torch.softmax(outputs, dim=-1)
                p = probs.cpu().numpy()
                p_score = np.max(p, -1)
                p_score_all = p           
                p = np.argmax(p, -1)
                splits = [p[sum(lengths[:i]):sum(lengths[:i + 1])] for i in range(len(lengths))]
                splits_score = [p_score_all[sum(lengths[:i]):sum(lengths[:i + 1])] for i in range(len(lengths))]
                splits_labels = [labels[sum(lengths[:i]):sum(lengths[:i + 1])] for i in range(len(lengths))]

                prediction.append(splits)
                gt.append(splits_labels)
                score.append(splits_score)

        prediction = sum(prediction, [])
        gt = sum(gt, [])
        score = sum(score, [])

        return score, prediction, gt
