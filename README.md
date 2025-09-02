# TransPHLA - A Transformer-based classifier for peptide-MHC binding prediction

This repository contains the code for my **MSc thesis**, which applies a **language learning model** to the challenging task of antigen presentation prediction. I conducted research on **peptide-HLA binding prediction** using a **transformer-based language learning model** to classify binders and non-binders. My work also involved exploring **generative models** to design new artificial peptides with high predicted binding affinity.

As part of this research, I performed extensive **hyperparameter tuning**, **attention analysis**, and **structure analysis** via clustering. I also used entropy-based diversity metrics to evaluate the generative model's output. All memory-intensive deep learning experiments were run in a **Linux HPC (High Performance Computing) environment**, requiring me to manage jobs across multiple nodes. The key technologies used in this project include **Python, PyTorch, NumPy, Matplotlib, scikit-learn, and the Hugging Face library**, important tools in **data science and statistics**.

---

## Overview

TransPHLA is a classification model that takes as input a peptide sequence and a corresponding MHC Class I allele (e.g., HLA-A\*02:01) and outputs a probability score indicating the likelihood of binding. The model's architecture consists of the following key components:

1. **Input Embedding:** The peptide and MHC sequences are first converted into a numerical representation.

2. **Transformer Encoder:** The embedded sequences are processed by a Transformer encoder, which utilizes a multi-head attention mechanism to learn the relationships between amino acid residues.

3. **Classification Head:** A final classification layer predicts the binding probability.

The model was trained on a comprehensive dataset of peptide-MHC binding data from the **Immune Epitope Database (IEDB)**, demonstrating strong performance and generalization to unseen data.

---

## Methodology and Model Architecture

The TransPHLA model is based on a Transformer encoder architecture, which uses a multi-head self-attention mechanism to process input sequences. This allows the model to weigh the importance of different amino acid residues when predicting binding.

### Input Embedding

Amino acid sequences for the peptide and the HLA allele are first converted into numerical vectors. These vectors are then summed with positional encodings to provide the model with information about the order of the amino acids.

### Multi-Head Attention

The core of the Transformer is the self-attention mechanism, which computes a score based on a query (Q), key (K), and value (V) matrix:

$$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

where $d_k$ is the dimension of the key vectors. Multi-head attention allows the model to attend to different parts of the sequence simultaneously.

### Cross-Entropy Loss

The model is trained as a binary classifier. During training, it minimizes the **cross-entropy loss** between the predicted binding probability and the ground truth label (binder or non-binder):

$$L = - \frac{1}{N} \sum_{i=1}^N [y_i \log(\hat{y_i}) + (1-y_i) \log(1-\hat{y_i})]$$

Here, $N$ is the number of samples, $y_i$ is the true label (1 for a binder, 0 for a non-binder), and $\hat{y_i}$ is the predicted binding probability.

---

## Performance

TransPHLA's performance was evaluated on a held-out test set, with the following results at a classification threshold of 0.5:

| **Set** | **AUC** | **Accuracy** | **MCC** | **F1** | **Specificity** | **Precision** | **Recall** | **AUPR** | 
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | 
| **Train** | 0.9999 | 0.9987 | 0.9974 | 0.9987 | 0.9986 | 0.9986 | 0.9988 | 0.9999 | 
| **Test** | 0.9682 | 0.9351 | 0.8652 | 0.9195 | 0.9425 | 0.9149 | 0.9242 | 0.977 | 

These results highlight the model's high predictive power in accurately classifying peptide-MHC binding pairs.

---

## Model Analysis

A detailed analysis of the TransPHLA model revealed that its cross-attention mechanism successfully grouped peptides based on their respective HLA alleles. This indicates that the model learned biologically relevant features, such as allele-specific binding motifs, which are crucial for accurate prediction.

### Visualization of Learned Features

To further investigate the model's ability to learn biologically relevant features, the `clustersUMAP.ipynb` Jupyter notebook was used to visualize the embeddings of the peptide-MHC pairs. This notebook employs **UMAP** to reduce the high-dimensional embeddings into a 2D space, allowing for the visual inspection of clusters formed by different HLA alleles. A similar visualization was also performed using **t-SNE**, which yielded comparable results, so it was not included in the final thesis to maintain focus. The **motivation** for this visualization was drawn from the **AntiBERTa** paper, and part of the code is based on the thesis by **Octave Malamoud**.

### Tokenizer

The `tokenizer/` folder contains `vocab.json`, a dictionary mapping each amino acid to a unique numerical token. This tokenizer is based on the work by **Leem et al. (2022)** from their self-supervised language model, AntiBERTa.

### **Evaluation and Results**

The `eval.py` script saves all of its output, including the performance plots, tables, and peptide-level scores, to the **`evals/default/`** directory. The summary table in the **`evals/default/model_performance_summary.png`** file and the corresponding CSV file summarize the key metrics for both the training and test sets, which are reflected in the table above. The generated performance plots, such as the ROC and Precision-Recall curves, are also saved in this directory.

In contrast, `evalByLength.py` saves its outputs to a separate folder: **`evals/evalByLength/`**. This directory contains:

* `training_metrics_by_length.png`: A bar chart showing the performance of the model on the training set for different peptide lengths.

* `test_metrics_by_length.png`: A bar chart showing the performance on the test set for different peptide lengths.

* `positive_binder_length_histogram.png`: A histogram showing the distribution of peptide lengths for all positive binders in the dataset.

---

## Hyperparameter Optimization

The optimal hyperparameters for the TransPHLA model were determined through a comprehensive grid search executed on the Imperial College London's HPC. This process was managed by two key scripts:

* **`runHyperParametricSearch.py`**: This script orchestrated the entire process by iterating through a predefined range of hyperparameters, such as the number of attention heads (`nHeads`), layers (`nLayers`), and dropout rate (`dropout`). For each unique combination, it submitted a job to the HPC's queuing system using the `qsub` command.

* **`run5Fold.py`**: This script represents a single job executed on the HPC. For a specific set of hyperparameters, it performs a **5-fold cross-validation** on the training data. For each fold, it trains a new instance of the model and evaluates it on the corresponding validation fold. It saves the best-performing model and its evaluation metrics for each fold, with the overall results being appended to the `hyperparametersMainModelFinal.csv` file.

The results of the hyperparameter sweeps are organized into dedicated folders for each run. The `losses` (e.g., `losses2`, `losses3`) and `modelsHyperparams` (e.g., `modelsHyperparams2`, `modelsHyperparams3`) directories contain the loss records and the saved model files, respectively, for the different hyperparameter combinations tested. The results from these sweeps were then analyzed using the `hyperparametricAnalysis.ipynb` Jupyter notebook. This analysis, which helped to identify the optimal model configuration, was partly done with code from PhD candidate Yinfei Yang. The `lossAnalysis.py` script was used to further analyze the convergence of the training loss. This systematic approach ensured that the final model was trained with the most effective configuration, maximizing its predictive performance.

---

## Codebase

* `eval.py` — Main evaluation script saving all outputs to `evals/default/`.
* `evalByLength.py` — Evaluation script for analyzing performance by peptide length.
* `runHyperParametricSearch.py` — Orchestrates hyperparameter grid search jobs on the HPC.
* `run5Fold.py` — A single job script that runs 5-fold cross-validation for a specific set of hyperparameters.
* `clustersUMAP.ipynb` — Jupyter notebook for visualizing peptide embeddings using UMAP.
* `hyperparametricAnalysis.ipynb` — Jupyter notebook for analyzing hyperparameter sweep results.
* `lossAnalysis.py` — Utilities for analyzing training/validation loss curves.

---

## Repository Structure

---

## Tools & Dependencies

* Python (>=3.8 recommended)
* PyTorch
* Hugging Face Transformers
* NumPy
* scikit-learn
* Matplotlib
* Linux HPC environment (for memory-intensive model training and generation)

(Include exact package versions in a `requirements.txt` or `environment.yml` for reproducibility.)

```
project_root/
├── tokenizer/                  # vocab.json and tokenizer utilities
├── data/              # processed CSVs (trainData, testData). Not in github
├── model/                      # saved models and checkpoints
├── modelsHyperparams/          # saved models from hyperparameter runs
├── evals/                      # evaluation outputs (ROC, PR, CSVs)
├── scripts/
├── train.py                # single-run training script (TransPHLA)
├── eval.py                 # evaluation script (saves to evals/default/)
├── evalByLength.py  # evaluation by peptide length (saves to evals/evalByLength/)
├── runHyperParametricSearch.py  # orchestrates HPC hyperparam sweep
├── run5Fold.py             # performs one hyperparam job: 5-fold CV
├── clustersUMAP.ipynb      # embedding visualizations
├── clusterstSNE.ipynb # embedding visualizations with t-SNE for experiment
├── hyperparametricAnalysis.ipynb
├── modelAndPerformances.py     # model classes, train/eval utility functions
├── lossAnalysis.py
├── requirements.txt
└── README.md
```

## References

[1] Chu, Y., et al. (2022). A transformer-based model to predict peptide–HLA class I binding and optimize mutated peptides for vaccine design. *Nature Machine Intelligence*, *4*(3), 300-311.

[2] Vaswani, A., et al. (2017). Attention is All You Need. In *Advances in Neural Information Processing Systems 30* (pp. 5998–6008).

[3] Imperial College Research Computing Service. (2022). *Imperial College Research Computing Service*. doi: 10.14469/hpc/2232. URL: https://doi.org/10.14469/hpc/2232

[4] Leem, J., et al. (2022). Deciphering the language of antibodies using self-supervised learning. *Patterns*, *3*(7), 100513. doi: 10.1016/j.patter.2022.100513.

[5] Malamoud, O. (2024). *Interpretability strategies for machine learning language models on antibodies sequences*. Imperial College London.

## Contact
felipepuiggarimedici@gmail.com https://www.linkedin.com/in/felipe-puiggari-medici-248855207/