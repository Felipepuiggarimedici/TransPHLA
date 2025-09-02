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
folder = "evals/new"
model_folder = "model/new"
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

colours = ['#9BC995', "#083D77", '#9A031E', '#C4B7CB', '#FC7753']
palette = sns.color_palette(colours, n_colors=15)

output_dir = folder
os.makedirs(output_dir, exist_ok=True)

d_model = 64
n_heads, n_layers, d_k, d_ff = (1,3,32, 768)  # Hyperparameters for the Transformer
dropout = 0.05
batch_size = 512
threshold = 0.5

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# --- Load data ---
data_test = pd.read_csv("data/testData/testData.csv")
pep_inputs_test, hla_inputs_test, labels_test = make_data(data_test)
loader_test = Data.DataLoader(MyDataSet(pep_inputs_test, hla_inputs_test, labels_test),
                              batch_size=batch_size, shuffle=False)
data_train = pd.read_csv("data/trainData/trainData.csv")
pep_inputs_train, hla_inputs_train, labels_train = make_data(data_train)
loader_train = Data.DataLoader(MyDataSet(pep_inputs_train, hla_inputs_train, labels_train),
                               batch_size=batch_size, shuffle=False)

# --- Load model ---
model = Transformer(d_model, d_k, n_layers, n_heads, d_ff, dropout=dropout).to(device)
model_path = os.path.join(model_folder, modelToEvaluate)
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} does not exist.")
print(f"Loading model from {model_path}")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
criterion = nn.CrossEntropyLoss()
all_results = []

# --- Evaluation function ---
def evaluate_and_save(set_name, loader, output_dir):
    print(f"\n--- Evaluating on {set_name} set ---")
    with torch.no_grad():
        ys_out, _, metrics = eval_step(model, loader, criterion, threshold, use_cuda)

    # Print metrics table
    print(performances_to_pd([metrics]))
    perf_df = performances_to_pd([metrics])
    perf_df.to_csv(os.path.join(output_dir, f"{set_name}_metrics.csv"), index=False)
    # Get predictions
    if isinstance(ys_out, dict):
        yTrue = np.asarray(ys_out.get("labels"))
        logits = ys_out.get("logits")
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().numpy()
        yScores = logits[:, 1] if logits.ndim > 1 else logits
        peptides = ys_out.get("peptides", [""]*len(yTrue))
    else:
        yTrue = np.asarray(ys_out[0])
        yScores = np.asarray(ys_out[2])
        peptides = [""]*len(yTrue)

    # enforce 1-D arrays
    yTrue = np.asarray(yTrue).reshape(-1)
    yScores = np.asarray(yScores).reshape(-1)

    perf_df = performances_to_pd([metrics])
    perf_df['Set'] = set_name.capitalize()
    all_results.append(perf_df)

    # Save CSV for individual set (optional, can remove if only want combined)
    perf_df.to_csv(os.path.join(output_dir, f"{set_name}_metrics.csv"), index=False)

    # ---- Compute ROC AUC & ROC curve ----
    auc_score = roc_auc_score(yTrue, yScores)
    fpr, tpr, thresholds = roc_curve(yTrue, yScores)
    print(f"AUC on {set_name}: {auc_score:.4f}")

    # Save ROC & AUC
    pd.DataFrame([{"Set": set_name, "AUC": auc_score}]).to_csv(os.path.join(output_dir, f"{set_name}_auc.csv"), index=False)
    pd.DataFrame({"FPR": fpr, "TPR": tpr, "Threshold": thresholds}).to_csv(os.path.join(output_dir, f"{set_name}_roc_curve.csv"), index=False)

    # ---- Compute PR curve & AUPR ----
    precision, recall, pr_thresholds = precision_recall_curve(yTrue, yScores)
    aupr_score = average_precision_score(yTrue, yScores)
    print(f"AUPR on {set_name}: {aupr_score:.4f}")

    # ---- Compute MCC ----
    yPred = (yScores >= threshold).astype(int)
    mcc_score = matthews_corrcoef(yTrue, yPred)
    print(f"MCC at threshold {threshold}: {mcc_score:.4f}")

    # Save MCC
    pd.DataFrame([{"Set": set_name, "MCC": mcc_score}]).to_csv(os.path.join(output_dir, f"{set_name}_mcc.csv"), index=False)
    # ---- Compute MCC on supported lengths only (8–13)
    if set_name == "train":
        all_peptides = data_train['peptide'].values
    elif set_name == "test":
        all_peptides = data_test['peptide'].values
    else:
        all_peptides = [""] * len(yTrue)

    # make sure lengths match yTrue
    all_peptides = all_peptides[:len(yTrue)]

    supported_mask = np.array([8 <= len(p) <= 13 for p in all_peptides])
    if supported_mask.any():
        mcc_supported = matthews_corrcoef(yTrue[supported_mask], yPred[supported_mask])
        print(f"MCC on supported peptide lengths (8–13): {mcc_supported:.4f}")
        pd.DataFrame([{"Set": set_name, "MCC_supported_8_13": mcc_supported}]) \
        .to_csv(os.path.join(output_dir, f"{set_name}_mcc_supported_8_13.csv"), index=False)
    else:
        print("No peptides in supported length range 8–13 for MCC computation.")


    # ---- Save peptide scores ----
    peptide_rows = [{"Peptide": p if p else f"idx_{i}", "Label": int(l), "LogProb": float(s)}
                    for i, (p, l, s) in enumerate(zip(peptides, yTrue, yScores))]
    pd.DataFrame(peptide_rows).to_csv(os.path.join(output_dir, f"{set_name}_peptide_scores.csv"), index=False)

    # ---- Plot ROC ----
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}", color=colours[1], linewidth=2.0, zorder=5, clip_on=False)
    ax.plot([0, 1], [0, 1], linestyle='--', color=colours[4], linewidth=0.9, zorder=1, clip_on=False)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve ({set_name.capitalize()} Set)")
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.margins(x=0, y=0)
    ax.set_axisbelow(False)
    leg = ax.legend(loc="lower right", frameon=True)
    leg.set_zorder(10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{set_name}_roc_curve.png"), dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close()

    # ---- Plot Precision-Recall ----
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.plot(recall, precision, label=f"AUPR = {aupr_score:.4f}", color=colours[2], linewidth=2.0, zorder=5, clip_on=False)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title(f"Precision–Recall Curve ({set_name.capitalize()} Set)")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.margins(x=0, y=0)
    ax2.set_axisbelow(False)
    leg2 = ax2.legend(loc="lower left", frameon=True)
    leg2.set_zorder(10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{set_name}_pr_curve.png"), dpi=300, bbox_inches='tight', pad_inches=0.03)
    plt.close()

# --- Run evaluation ---
evaluate_and_save("train", loader_train, output_dir)
evaluate_and_save("test", loader_test, output_dir)

# --- Build single summary table ---
summary_df = pd.concat(all_results, ignore_index=True)

# --- Build single summary table with only first row per set ---
summary_df = pd.concat([df.iloc[0:1] for df in all_results], ignore_index=True)

# Rename columns for more readable headers
summary_df = summary_df.rename(columns={
    "roc_auc": "AUC",
    "accuracy": "Accuracy",
    "mcc": "MCC",
    "f1": "F1",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "precision": "Precision",
    "recall": "Recall",
    "aupr": "AUPR"
})

# Round for readability
summary_df_rounded = summary_df.round(4)

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")

table = ax.table(
    cellText=summary_df_rounded.drop(columns=["Set"]).values,
    colLabels=summary_df_rounded.drop(columns=["Set"]).columns,
    rowLabels=summary_df_rounded["Set"],
    cellLoc='center',
    rowLoc='center',
    loc='center'
)

# Styling
table.auto_set_font_size(False)
table.set_fontsize(12)
table.auto_set_column_width(col=list(range(len(summary_df_rounded.columns)-1)))

for (i, j), cell in table.get_celld().items():
    if i == 0:  # header row
        cell.set_text_props(weight='bold', color='black')
        cell.set_facecolor('#E0E0E0')  # light gray
        cell.set_edgecolor('black')
        cell.set_linewidth(1.2)
    else:  # body rows
        cell.set_facecolor('white')
        cell.set_text_props(color='black')
        if j == -1:  # row labels (Train/Test)
            cell.set_text_props(weight='bold')
        cell.set_edgecolor('black')
        cell.set_linewidth(0.6)

# Remove vertical gridlines
for key, cell in table.get_celld().items():
    if key[1] != -1:
        cell.visible_edges = "horizontal"

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "model_performance_summary.png"),
            dpi=300, bbox_inches='tight')
plt.show()