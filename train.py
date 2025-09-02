'''
Code inspired by TransPHLA study [1] with added features and modifications
'''
from modelAndPerformances import *
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.utils.data as Data
# Ensure working directory is script location
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
pep_max_len = 9  # peptide; enc_input max sequence length
hla_max_len = 34  # hla; dec_input (=dec_output) max sequence length
tgt_len = pep_max_len + hla_max_len

number = ""  
folder = "model/"
output_dir = os.path.join(folder, number)
os.makedirs(output_dir, exist_ok=True)

with open("tokenizer/vocab.json", "r") as f:
    vocab = json.load(f)
vocab_size = len(vocab)

# Transformer Parameters
d_model = 64  # Embedding Size
patienceFull = 10  # Patience for early stopping

n_heads, n_layers, d_k, d_ff = (1,3,32, 768)  # Hyperparameters for the Transformer
dropout = 0.05
batch_size = 512  # Batch size for training
epochs = 300
threshold = 0.5

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
results = []

data = pd.read_csv('data/trainData/trainData.csv')
pep_inputs, hla_inputs, labels = make_data(data)
dataset = MyDataSet(pep_inputs, hla_inputs, labels)
loader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

train_data, train_pep_inputs, train_hla_inputs, train_labels, train_loader = data, pep_inputs, hla_inputs, labels, loader

model = Transformer(d_model, d_k, n_layers, n_heads, d_ff, dropout=dropout).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

best_loss = float('inf')
patience_counter = 0
avgLosses = np.zeros(epochs)

print("Beginning training")

for epoch in range(1, epochs + 1):
    print("Epoch: ", epoch)
    _, loss_train_list, metrics_train, time_train_ep = train_step(
        model, train_loader, -1, epoch, epochs, criterion, optimizer, threshold, use_cuda
    )

    avg_loss = loss_train_list.item()
    avgLosses[epoch - 1] = avg_loss
    print(f"Epoch {epoch}, Training Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        torch.save(model.state_dict(), os.path.join(output_dir, 'best_model_final.pt'))
        print("Model improved. Saved.")
    else:
        patience_counter += 1
        print(f"No improvement. Patience counter: {patience_counter}/{patienceFull}")

    if patience_counter >= patienceFull:
        print("Early stopping triggered.")
        break

np.save(os.path.join(output_dir, 'avg_losses.npy'), avgLosses)
print("Training finished. Outputs saved to:", output_dir)