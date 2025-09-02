import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import argparse
from collections import Counter
import numpy as np
# Ensure working directory is script location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
from modelAndPerformances import *
from torch.utils.data import TensorDataset, DataLoader
# Ensure working directory is script location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
# === Argument Parsing ===

parser = argparse.ArgumentParser(description="Transformer Hyperparameter Sweep - Five-Fold Training")
parser.add_argument("-nHeads", type=int, required=True, help="Number of heads")
parser.add_argument("-nLayers", type=int, required=True, help="Number of layers")
parser.add_argument("-d_k", type=int, required=True, help="Dimension d_k")
parser.add_argument("-d_ff", type=int, required=True, help="Feedforward dimension")
parser.add_argument("-dropout", type=float, required=True, help="Dropout rate")
args = parser.parse_args()

nHeads = args.nHeads
nLayers = args.nLayers
d_k = args.d_k
d_ff = args.d_ff
dropout = args.dropout
# === Constants ===
pep_max_len = 46
hla_max_len = 34
tgt_len = pep_max_len + hla_max_len
d_model = 64
batch_size_mini = 1024
epochs = 100
threshold = 0.5
patienceMini = 3
num_folds = 5

# === Device ===
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# === Load vocab ===
with open("tokenizer/vocab.json", "r") as f:
    vocab = json.load(f)
vocab_size = len(vocab)

# === Folder setup ===
os.makedirs("losses3", exist_ok=True)
os.makedirs("modelsHyperparams3", exist_ok=True)

# === Five-Fold Training ===
all_fold_results = []
fold_loss_records = []

for val_fold in range(num_folds):
    train_folds = [i for i in range(num_folds) if i != val_fold]
    
    # Combine train data loaders for all train_folds
    train_data_all, train_pep_inputs_all, train_hla_inputs_all, train_labels_all = [], [], [], []
    for fold in train_folds:
        data, pep_inputs, hla_inputs, labels, loader = data_with_loader(type_='train', fold=fold, batch_size=batch_size_mini)
        train_data_all.append(data)
        train_pep_inputs_all.append(pep_inputs)
        train_hla_inputs_all.append(hla_inputs)
        train_labels_all.append(labels)

    train_data = pd.concat(train_data_all, ignore_index=True)
    train_pep_inputs = torch.cat(train_pep_inputs_all, dim=0)
    train_hla_inputs = torch.cat(train_hla_inputs_all, dim=0)
    train_labels = torch.cat(train_labels_all, dim=0)

    train_dataset = TensorDataset(train_pep_inputs, train_hla_inputs, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size_mini, shuffle=True)

    # Validation data loader
    val_data, val_pep_inputs, val_hla_inputs, val_labels, val_loader = data_with_loader(type_='val', fold=val_fold, batch_size=batch_size_mini)

    print(f"===== Fold {val_fold}: Train folds {train_folds}, Val fold {val_fold} =====")
    print(f"Train label distribution: {Counter(train_data.label)}")
    print(f"Validation label distribution: {Counter(val_data.label)}")

    # === Model ===
    model = Transformer(d_model, d_k, nLayers, nHeads, d_ff, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # === Training ===
    best_val_loss = float('inf')
    patienceCounter = 0
    best_model_state = None
    fold_losses = []

    for epoch in range(1, epochs + 1):
        _, train_loss, _, _ = train_step(model, train_loader, val_fold, epoch, epochs, criterion, optimizer, threshold, use_cuda)
        _, val_loss, val_metrics = eval_step(model, val_loader, criterion, threshold, use_cuda=use_cuda)

        fold_losses.append({
            'fold': val_fold,
            'epoch': epoch,
            'train_loss': float(train_loss) if isinstance(train_loss, torch.Tensor) else train_loss,
            'val_loss': float(val_loss) if isinstance(val_loss, torch.Tensor) else val_loss
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = model.state_dict()
            patienceCounter = 0
        else:
            patienceCounter += 1

        if patienceCounter >= patienceMini:
            print(f"Early stopping at epoch {epoch} for fold {val_fold}")
            break

    fold_loss_records.extend(fold_losses)

    # === Save best model for fold ===
    model_filename = f"modelsHyperparams3/model_n{nHeads}_l{nLayers}_dk{d_k}_dff{d_ff}_do{dropout:.2f}_fold{val_fold}.pt"
    if best_model_state is not None:
        torch.save(best_model_state, model_filename)
        print(f"Saved best model for fold {val_fold} to {model_filename}")

    # === Evaluation metrics for fold ===
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.eval()
        _, _, val_metrics = eval_step(model, val_loader, criterion, threshold, use_cuda=use_cuda)
        all_fold_results.append({
            'fold': val_fold,
            'n_heads': nHeads,
            'n_layers': nLayers,
            'd_k': d_k,
            'd_ff': d_ff,
            'dropout': dropout,
            'roc_auc': val_metrics[8],
            'accuracy': val_metrics[1],
            'mcc': val_metrics[2],
            'f1': val_metrics[3],
            'sensitivity': val_metrics[4],
            'specificity': val_metrics[5],
            'precision': val_metrics[6],
            'recall': val_metrics[7],
            'aupr': val_metrics[0]
        })

# === Save all fold losses ===
loss_filename = f"losses3/losses_n{nHeads}_l{nLayers}_dk{d_k}_dff{d_ff}_do{dropout:.2f}_fiveFold.csv"
pd.DataFrame(fold_loss_records).to_csv(loss_filename, index=False)
print(f"Saved all fold losses to {loss_filename}")

# === Save all fold metrics ===
results_df = pd.DataFrame(all_fold_results)
csv_path = 'hyperparametersMainModelFinal.csv'
expected_columns = [
    'n_heads', 'n_layers', 'd_k', 'd_ff', 'dropout',
    'roc_auc', 'accuracy', 'mcc', 'f1',
    'sensitivity', 'specificity', 'precision',
    'recall', 'aupr'
]

if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
    pd.DataFrame(columns=expected_columns).to_csv(csv_path, index=False)
    print(f"Created new CSV with headers at {csv_path}")

existing_df = pd.read_csv(csv_path)
existing_df.set_index(['n_heads', 'n_layers', 'd_k', 'd_ff'], inplace=True)
results_df.set_index(['n_heads', 'n_layers', 'd_k', 'd_ff'], inplace=True)

existing_df.update(results_df)
combined_df = pd.concat([
    existing_df,
    results_df.loc[~results_df.index.isin(existing_df.index)]
])
combined_df.reset_index(inplace=True)
combined_df = combined_df[combined_df['d_ff'].notna()]
combined_df.to_csv(csv_path, index=False)
print(f"Updated and saved results in {csv_path}")
print("----- Finished Five-Fold Training -----")
