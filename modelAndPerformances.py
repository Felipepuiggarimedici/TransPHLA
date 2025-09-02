'''
Code by original TransPHLA study [1] with added features and modifications
'''
import math
from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import time
import datetime
import random
random.seed(1234)

from numpy import interp
import warnings
warnings.filterwarnings("ignore")
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

from collections import Counter
from functools import reduce
from tqdm import tqdm, trange
from copy import deepcopy

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, auc
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

seed = 19961231
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

pep_max_len = 46 # peptide; enc_input max sequence length
hla_max_len = 34 # hla; dec_input(=dec_output) max sequence length
tgt_len = pep_max_len + hla_max_len
pep_max_len, hla_max_len
import json

with open("tokenizer/vocab.json", "r") as f:
    vocab = json.load(f)

vocab_size = len(vocab)

def data_with_loader(type_='train', fold=0, batch_size=1024):
    if type_ == 'train':
        path = f"data/trainDataFolds/train_data_fold_{fold}.csv"
    elif type_ == 'val':
        path = f"data/trainDataFolds/val_data_fold_{fold}.csv"
    else:
        raise ValueError("type_ must be 'train' or 'val'.")
    
    data = pd.read_csv(path)
    pep_inputs, hla_inputs, labels = make_data(data)
    dataset = MyDataSet(pep_inputs, hla_inputs, labels)
    loader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=(type_ == 'train'), num_workers=0)

    return data, pep_inputs, hla_inputs, labels, loader

def make_data(data, HLAFlag = False):
    'If flag is set to True the hlaLabels will be returned as well'
    pep_inputs, hla_inputs, labels, hlaLabels = [], [], [], []
    for pep, hla, label, hlaLabel in zip(data.peptide, data.HLA_sequence, data.label, data.HLA):
        if pd.isna(hla):
            continue
        pep, hla = pep.ljust(pep_max_len, '-'), hla.ljust(hla_max_len, '-')
        pep_input = [[vocab[n] for n in pep]] # [[1, 2, 3, 4, 0], [1, 2, 3, 5, 0]]
        hla_input = [[vocab[n] for n in hla]]
        pep_inputs.extend(pep_input)
        hla_inputs.extend(hla_input)
        labels.append(label)
        hlaLabels.append(hlaLabel)
    if HLAFlag:
        return torch.LongTensor(pep_inputs), torch.LongTensor(hla_inputs), torch.LongTensor(labels),  np.array(hlaLabels)
    else:
        return torch.LongTensor(pep_inputs), torch.LongTensor(hla_inputs), torch.LongTensor(labels)

class MyDataSet(Data.Dataset):
    def __init__(self, pep_inputs, hla_inputs, labels):
        super(MyDataSet, self).__init__()
        self.pep_inputs = pep_inputs
        self.hla_inputs = hla_inputs
        self.labels = labels

    def __len__(self): # 样本数
        return self.pep_inputs.shape[0] # 改成hla_inputs也可以哦！

    def __getitem__(self, idx):
        return self.pep_inputs[idx], self.hla_inputs[idx], self.labels[idx]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=0):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)  # added dropout layer

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)  # [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)  # mask out irrelevant positions
        
        attn = nn.Softmax(dim=-1)(scores)
        attn = self.dropout(attn)  # apply dropout here
        
        context = torch.matmul(attn, V)  # [batch_size, n_heads, len_q, d_v]
        return context, attn

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, dropout=0):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)          # dropout layer for output
        self.d_model = d_model
        self.d_v = d_v
        self.d_k = d_k
        self.n_heads = n_heads
        self.scaled_dot_attn = ScaledDotProductAttention(d_k, dropout)  # reuse with dropout

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        
        Q = self.W_Q(input_Q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)  # [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, self.n_heads, self.d_v).transpose(1, 2)  # [batch_size, n_heads, len_v, d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_heads, 1, 1).to(Q.device)  # [batch_size, n_heads, seq_len, seq_len]

        context, attn = self.scaled_dot_attn(Q, K, V, attn_mask)  # apply scaled dot product attention with dropout inside
        
        context = context.transpose(1, 2).reshape(batch_size, -1, self.n_heads * self.d_v)  # [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)  # [batch_size, len_q, d_model]
        output = self.dropout(output)  # dropout after final linear
        
        return self.norm(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0):
        super(PoswiseFeedForwardNet, self).__init__()
        self.d_model = d_model
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),       # dropout added here
            nn.Linear(d_ff, d_model, bias=False),
        )
        self.norm = nn.LayerNorm(self.d_model)

    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return self.norm(output + residual)  # [batch_size, seq_len, d_model]

# In[19]:

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads, d_ff, dropout=0):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, d_k, d_v, n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)

    def forward(self, enc_inputs, enc_self_attn_mask):
        '''
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]
        '''
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class Encoder(nn.Module):
    def __init__(self, d_model, n_layers, d_k, n_heads, d_ff, dropout=0):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, dropout)  # Optional: if your positional encoding supports dropout
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, d_k, d_k, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.src_emb(enc_inputs)  # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)  # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns

# ### Decoder
class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_k, n_heads, d_ff, dropout=0):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention(d_model, d_k, d_k, n_heads, dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, dropout)

    def forward(self, dec_inputs, dec_self_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        '''
        dec_outputs, dec_self_attn = self.dec_self_attn(
            dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask
        )
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs, dec_self_attn


class Decoder(nn.Module):
    def __init__(self, d_model, n_layers, d_k, n_heads, d_ff, tgt_len, dropout=0):
        super(Decoder, self).__init__()
        self.pos_emb = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, d_k, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.tgt_len = tgt_len

    def forward(self, dec_inputs):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        '''
        dec_outputs = self.pos_emb(dec_inputs.transpose(0, 1)).transpose(0, 1)
        dec_self_attn_pad_mask = torch.zeros(
            (dec_inputs.shape[0], self.tgt_len, self.tgt_len), dtype=torch.bool, device=dec_inputs.device
        )

        dec_self_attns = []
        for layer in self.layers:
            dec_outputs, dec_self_attn = layer(dec_outputs, dec_self_attn_pad_mask)
            dec_self_attns.append(dec_self_attn)

        return dec_outputs, dec_self_attns
    
class Transformer(nn.Module):
    def __init__(self, d_model, d_k, n_layers, n_heads, d_ff, tgt_len=tgt_len, dropout=0.1):
        super(Transformer, self).__init__()
        self.pep_encoder = Encoder(d_model, n_layers, d_k, n_heads, d_ff, dropout= dropout)
        self.hla_encoder = Encoder(d_model, n_layers, d_k, n_heads, d_ff, dropout= dropout)
        self.decoder = Decoder(d_model, n_layers, d_k, n_heads, d_ff, tgt_len, dropout= dropout)
        self.tgt_len = tgt_len
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.projection = nn.Sequential(
            nn.Linear(tgt_len * d_model, 256),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Dropout(dropout),
            nn.Linear(64, 2)
        )

    def forward(self, pep_inputs, hla_inputs):
        '''
        pep_inputs: [batch_size, pep_len]
        hla_inputs: [batch_size, hla_len]
        '''
        pep_enc_outputs, pep_enc_self_attns = self.pep_encoder(pep_inputs)
        hla_enc_outputs, hla_enc_self_attns = self.hla_encoder(hla_inputs)
        enc_outputs = torch.cat((pep_enc_outputs, hla_enc_outputs), 1)

        dec_outputs, dec_self_attns = self.decoder(enc_outputs)
        dec_outputs = dec_outputs.view(dec_outputs.shape[0], -1)
        dec_logits = self.projection(dec_outputs)

        # Return all outputs and all attention weights
        return dec_logits.view(-1, dec_logits.size(-1)), pep_enc_self_attns, hla_enc_self_attns, dec_self_attns

def performances(y_true, y_pred, y_prob, print_ = True):
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels = [0, 1]).ravel().tolist()
    accuracy = (tp+tn)/(tn+fp+fn+tp)
    denom = np.sqrt(np.float64((tp+fn)*(tn+fp)*(tp+fp)*(tn+fn)))
    if denom > 0:
        mcc = ((tp*tn) - (fn*fp)) / denom
        print("MCC Error:", mcc)
    else: 
        print("MCC denom is 0")
        mcc= np.nan
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    
    try:
        recall = tp / (tp+fn)
    except:
        recall = np.nan
        
    try:
        precision = tp / (tp+fp)
    except:
        precision = np.nan
        
    try: 
        f1 = 2*precision*recall / (precision+recall)
    except:
        f1 = np.nan
        
    roc_auc = roc_auc_score(y_true, y_prob)
    prec, reca, _ = precision_recall_curve(y_true, y_prob)
    aupr = auc(reca, prec)
    
    if print_:
        print('tn = {}, fp = {}, fn = {}, tp = {}'.format(tn, fp, fn, tp))
        print('y_pred: 0 = {} | 1 = {}'.format(Counter(y_pred)[0], Counter(y_pred)[1]))
        print('y_true: 0 = {} | 1 = {}'.format(Counter(y_true)[0], Counter(y_true)[1]))
        print('auc={:.4f}|sensitivity={:.4f}|specificity={:.4f}|acc={:.4f}|mcc={:.4f}'.format(roc_auc, sensitivity, specificity, accuracy, mcc))
        print('precision={:.4f}|recall={:.4f}|f1={:.4f}|aupr={:.4f}'.format(precision, recall, f1, aupr))
    
    return (aupr, accuracy, mcc, f1, sensitivity, specificity, precision, recall, roc_auc)

def transfer(y_prob, threshold = 0.5):
    return np.array([[0, 1][x > threshold] for x in y_prob])

f_mean = lambda l: sum(l)/len(l)


def performances_to_pd(performances_list):
    metrics_name = ['roc_auc', 'accuracy', 'mcc', 'f1', 'sensitivity', 'specificity', 'precision', 'recall', 'aupr']

    performances_pd = pd.DataFrame(performances_list, columns = metrics_name)
    performances_pd.loc['mean'] = performances_pd.mean(axis = 0)
    performances_pd.loc['std'] = performances_pd.std(axis = 0)
    
    return performances_pd


# In[7]:


def train_step(model, train_loader, fold, epoch, epochs, criterion, optimizer, threshold = 0.5, use_cuda = True):
    device = torch.device("cuda" if use_cuda else "cpu")
    
    time_train_ep = 0
    model.train()
    y_true_train_list, y_prob_train_list = [], []
    loss_train_list, dec_attns_train_list = [], []
    for train_pep_inputs, train_hla_inputs, train_labels in tqdm(train_loader):
        '''
        pep_inputs: [batch_size, pep_len]
        hla_inputs: [batch_size, hla_len]
        train_outputs: [batch_size, 2]
        '''
        train_pep_inputs, train_hla_inputs, train_labels = train_pep_inputs.to(device), train_hla_inputs.to(device), train_labels.to(device)

        t1 = time.time()
        train_outputs, _, _, train_dec_self_attns = model(train_pep_inputs, 
                                                                                                        train_hla_inputs)
        train_loss = criterion(train_outputs, train_labels)
        time_train_ep += time.time() - t1

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        y_true_train = train_labels.cpu().numpy()
        y_prob_train = nn.Softmax(dim = 1)(train_outputs)[:, 1].cpu().detach().numpy()
        
        y_true_train_list.extend(y_true_train)
        y_prob_train_list.extend(y_prob_train)
        loss_train_list.append(train_loss)
#         dec_attns_train_list.append(train_dec_self_attns)
        
    y_pred_train_list = transfer(y_prob_train_list, threshold)
    ys_train = (y_true_train_list, y_pred_train_list, y_prob_train_list)
    if fold not in [0,1,2,3,4]:
        print('****Train (Ep avg): Epoch-{}/{} | Loss = {:.4f} | Time = {:.4f} sec'.format(epoch, epochs, f_mean(loss_train_list), time_train_ep))
    else:
        print('Fold-{}****Train (Ep avg): Epoch-{}/{} | Loss = {:.4f} | Time = {:.4f} sec'.format(fold, epoch, epochs, f_mean(loss_train_list), time_train_ep))
    metrics_train = performances(y_true_train_list, y_pred_train_list, y_prob_train_list, print_ = True)
    return ys_train,f_mean(loss_train_list), metrics_train, time_train_ep#, dec_attns_train_list

def eval_step(model, val_loader, criterion, threshold=0.5, use_cuda=True, returnAttentionData=False):
    device = torch.device("cuda" if use_cuda else "cpu")
    model.eval()

    loss_val_list = []
    dec_attns_val_list = [] if returnAttentionData else None
    y_true_val_list = []
    y_prob_val_list = []

    with torch.no_grad():
        for batch_idx, (val_pep_inputs, val_hla_inputs, val_labels) in enumerate(tqdm(val_loader)):
            # move to device
            val_pep_inputs = val_pep_inputs.to(device)
            val_hla_inputs = val_hla_inputs.to(device)
            val_labels = val_labels.to(device)

            # forward
            val_outputs, _, _, val_dec_self_attns = model(val_pep_inputs, val_hla_inputs)
            val_loss = criterion(val_outputs, val_labels)

            # probs to CPU numpy
            y_prob_val = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()
            y_true_val = val_labels.cpu().numpy()

            # collect scalars / arrays (no GPU tensors saved)
            loss_val_list.append(float(val_loss.item()))
            y_true_val_list.extend(y_true_val)
            y_prob_val_list.extend(y_prob_val)

            # only store attention if explicitly requested; move to CPU and detach
            if returnAttentionData:
                # Convert any nested structure of tensors to numpy on CPU
                def to_cpu_numpy(x):
                    if isinstance(x, torch.Tensor):
                        return x.detach().cpu().numpy()
                    elif isinstance(x, (list, tuple)):
                        return [to_cpu_numpy(xx) for xx in x]
                    elif isinstance(x, dict):
                        return {k: to_cpu_numpy(v) for k, v in x.items()}
                    else:
                        return x
                dec_attns_val_list.append(to_cpu_numpy(val_dec_self_attns))

            # periodically free cached memory to avoid fragmentation issues
            if use_cuda and (batch_idx % 50 == 0):
                torch.cuda.empty_cache()

    # predictions and metrics
    y_pred_val_list = transfer(y_prob_val_list, threshold)
    ys_val = (y_true_val_list, y_pred_val_list, y_prob_val_list)
    metrics_val = performances(y_true_val_list, y_pred_val_list, y_prob_val_list, print_ = True)

    if returnAttentionData:
        return ys_val, np.mean(loss_val_list), metrics_val, dec_attns_val_list
    else:
        return ys_val, np.mean(loss_val_list), metrics_val
