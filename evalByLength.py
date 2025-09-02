import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, matthews_corrcoef
from modelAndPerformances import Transformer, eval_step, make_data, MyDataSet, performances_to_pd

# --- Settings ---
folder = "evals/evalByLength"
model_folder = "model"
modelToEvaluate = "best_model_final.pt"

seed = 6
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 20
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=BIGGER_SIZE)
plt.rc('axes', labelsize=MEDIUM_SIZE)
plt.rc('xtick', labelsize=SMALL_SIZE)
plt.rc('ytick', labelsize=SMALL_SIZE)
plt.rc('legend', fontsize=MEDIUM_SIZE)
plt.rc('figure', titlesize=BIGGER_SIZE)

# Custom color palette
colors = ['#9BC995', "#083D77", '#9A031E', '#C4B7CB', '#FC7753']
palette = sns.color_palette(colors, n_colors=15)

output_dir = folder
os.makedirs(output_dir, exist_ok=True)

d_model = 64
n_heads, n_layers, d_k, d_ff = (1, 3, 32, 768)
dropout = 0.05
batch_size = 512
threshold = 0.5

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# --- Load full data once ---
data_test = pd.read_csv("data/testData/testData.csv")
data_train = pd.read_csv("data/trainData/trainData.csv")

# --- Load model once ---
model = Transformer(d_model, d_k, n_layers, n_heads, d_ff, dropout=dropout).to(device)
model_path = os.path.join(model_folder, modelToEvaluate)
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} does not exist.")
print(f"Loading model from {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
criterion = nn.CrossEntropyLoss()

# --- Dictionaries to store results for plotting ---
train_metrics_by_length = {metric: [] for metric in ["AUC", "AUPR", "MCC", "F1"]}
test_metrics_by_length = {metric: [] for metric in ["AUC", "AUPR", "MCC", "F1"]}
lengths_to_evaluate = range(8, 14)

# --- Loop through each peptide length ---
for length in lengths_to_evaluate:
    print(f"\n--- Evaluating on peptides of length {length} ---")
    
    # Filter data by length
    data_test_len = data_test[data_test['peptide'].str.len() == length]
    data_train_len = data_train[data_train['peptide'].str.len() == length]
    
    # Skip if no data for this length
    if data_test_len.empty and data_train_len.empty:
        print(f"No data for length {length}. Skipping.")
        continue

    # Create DataLoaders for the filtered data
    if not data_train_len.empty:
        pep_inputs_train, hla_inputs_train, labels_train = make_data(data_train_len)
        loader_train = Data.DataLoader(MyDataSet(pep_inputs_train, hla_inputs_train, labels_train),
                                     batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            _, _, metrics_tuple = eval_step(model, loader_train, criterion, threshold, use_cuda)
        
        # Store training metrics from the tuple
        # The performances function returns (aupr, accuracy, mcc, f1, sensitivity, specificity, precision, recall, roc_auc)
        train_metrics_by_length["AUPR"].append(metrics_tuple[0])
        train_metrics_by_length["MCC"].append(metrics_tuple[2])
        train_metrics_by_length["F1"].append(metrics_tuple[3])
        train_metrics_by_length["AUC"].append(metrics_tuple[8])


    if not data_test_len.empty:
        pep_inputs_test, hla_inputs_test, labels_test = make_data(data_test_len)
        loader_test = Data.DataLoader(MyDataSet(pep_inputs_test, hla_inputs_test, labels_test),
                                     batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            _, _, metrics_tuple = eval_step(model, loader_test, criterion, threshold, use_cuda)
        
        # Store test metrics from the tuple
        # The performances function returns (aupr, accuracy, mcc, f1, sensitivity, specificity, precision, recall, roc_auc)
        test_metrics_by_length["AUPR"].append(metrics_tuple[0])
        test_metrics_by_length["MCC"].append(metrics_tuple[2])
        test_metrics_by_length["F1"].append(metrics_tuple[3])
        test_metrics_by_length["AUC"].append(metrics_tuple[8])


# --- Plotting function for bar plots ---
def plot_metrics(metrics_dict, lengths, set_name):
    fig, ax = plt.subplots(figsize=(12, 5.6))
    x = np.arange(len(lengths))
    width = 0.2
    
    # Get the correct lengths that were actually evaluated
    evaluated_lengths = [l for l in lengths_to_evaluate if not data_train[data_train['peptide'].str.len() == l].empty or not data_test[data_test['peptide'].str.len() == l].empty]
    x = np.arange(len(evaluated_lengths))

    for i, (metric, values) in enumerate(metrics_dict.items()):
        if values: # Only plot if there are values for this metric
            bars = ax.bar(x + i * width, values, width, label=metric, color=colors[i % len(colors)])
            
            # Label the values on top of the bars, rounded to 2 decimal places
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=10)

    ax.set_title(f'{set_name} Set Metrics by Peptide Length', fontsize=BIGGER_SIZE)
    ax.set_xlabel('Peptide Length', fontsize=MEDIUM_SIZE)
    ax.set_ylabel('Score', fontsize=MEDIUM_SIZE)
    ax.set_xticks(x + width * (len(metrics_dict) - 1) / 2)
    ax.set_xticklabels(evaluated_lengths)
    ax.set_ylim(0, 1.1)
    ax.legend(title='Metrics', fontsize=MEDIUM_SIZE)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{set_name.lower()}_metrics_by_length.png'), dpi=300)
    plt.close()

# --- New plotting function for histogram ---
def plot_positive_binder_histogram(data_train, data_test):
    """
    Plots a histogram of peptide lengths for all positive binders (Label = 1) 
    for lengths 8 to 13.
    """
    # Combine data and filter for positive binders
    all_data = pd.concat([data_train, data_test])
    positive_binders = all_data[all_data['label'] == 1]
    
    # Calculate peptide lengths and filter for the desired range
    peptide_lengths = positive_binders['peptide'].str.len()
    peptide_lengths = peptide_lengths[(peptide_lengths >= 8) & (peptide_lengths <= 13)]
    
    # Plot histogram
    fig, ax = plt.subplots(figsize=(12, 5.6))
    
    # Set bins to cover the entire range from 8 to 13
    bins = np.arange(7.5, 14.5, 1)
    
    n, bins, patches = ax.hist(peptide_lengths, bins=bins, color=colors[1], edgecolor='black', rwidth=0.85)
    
    # Add labels on top of the bars
    for i, num in enumerate(n):
        if num > 0:
            ax.annotate(f'{int(num)}',
                        xy=(bins[i] + 0.5, num),
                        xytext=(0, 3),
                        textcoords='offset points',
                        ha='center', va='bottom', fontsize=12)

    ax.set_title('Distribution of Positive Binder Peptide Lengths (8-13)', fontsize=BIGGER_SIZE)
    ax.set_xlabel('Peptide Length', fontsize=MEDIUM_SIZE)
    ax.set_ylabel('Number of Peptides', fontsize=MEDIUM_SIZE)
    ax.set_xticks(range(8, 14))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'positive_binder_length_histogram.png'), dpi=300)
    plt.close()
    print("Histogram of positive binder peptide lengths created.")

# --- Create the bar plots ---
plot_metrics(train_metrics_by_length, list(lengths_to_evaluate), 'Training')
plot_metrics(test_metrics_by_length, list(lengths_to_evaluate), 'Test')

# --- Create the new histogram plot ---
plot_positive_binder_histogram(data_train, data_test)

print("\nEvaluation by length and plotting complete.")