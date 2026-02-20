# =========================================================================================
# Multi-Label Toxic Comment Filtering System
# ARCHITECTURE: RoBERTa-base + Focal Loss + Multi-Sample Dropout
# =========================================================================================

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
import warnings


warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# =========================================================================================
# 1. CONFIGURATION & HYPERPARAMETERS 
# =========================================================================================
class Config:
    DATA_PATH = '/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv'
    TEST_DATA_PATH = '/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv'
    TEST_LABELS_PATH = '/kaggle/input/jigsaw-toxic-comment-classification-challenge/test_labels.csv'
    SAVE_DIR = './saved_toxic_roberta_model' 
    
    # Model Settings
    MODEL_NAME = 'roberta-base'
    MAX_LEN = 128          
    TRAIN_BATCH_SIZE = 32  
    VALID_BATCH_SIZE = 64
    EPOCHS = 2             
    LEARNING_RATE = 2e-5
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    
   
    CUSTOM_THRESHOLDS = {
        'toxic': 0.50, 'severe_toxic': 0.50, 'obscene': 0.50, 
        'threat': 0.50, 'insult': 0.50, 'identity_hate': 0.50
    }
    
    SEVERITY_WEIGHTS = {
        'toxic': 1, 'severe_toxic': 5, 'obscene': 2, 
        'threat': 5, 'insult': 1, 'identity_hate': 4
    }

print(f"Hardware utilized: {Config.DEVICE}")

# =========================================================================================
# 2. CUSTOM MODULES (Focal Loss & Multi-Sample Dropout)
# =========================================================================================

class MultiLabelFocalLoss(nn.Module):
    """
    Down-weights easy examples and focuses gradient updates on rare classes.
    Formula: FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(MultiLabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_weight * bce_loss
        return focal_loss.mean()

class RobertaMultiSampleDropoutHead(nn.Module):
    """
    Creates an internal ensemble by passing features through multiple 
    dropout masks and averaging the classification logits.
    """
    def __init__(self, hidden_size, num_labels, num_samples=5, dropout_rate=0.2):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.num_samples = num_samples
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_samples)])
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, features):
        x = features[:, 0, :] 
        x = self.dense(x)
        x = torch.tanh(x)
        
        logits = torch.mean(
            torch.stack([self.out_proj(dropout(x)) for dropout in self.dropouts], dim=0),
            dim=0
        )
        return logits

# =========================================================================================
# 3. DATA PREPROCESSING MODULE
# =========================================================================================

def clean_text(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r'[^\x00-\x7F]+', '', text) 
    return text.strip()

class ToxicDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.text = dataframe.comment_text
        self.targets = dataframe[Config.LABELS].values

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = " ".join(str(self.text[index]).split())
        inputs = self.tokenizer.encode_plus(
            text, None, add_special_tokens=True, max_length=Config.MAX_LEN,
            padding='max_length', truncation=True, return_token_type_ids=False
        )
        return {
            'ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

# =========================================================================================
# 4. CORE TRAINING & VALIDATION PIPELINE
# =========================================================================================

def train(epoch, model, dataloader, optimizer, criterion, scheduler):
    model.train()
    fin_loss = 0
    print(f"\n--> Training Epoch {epoch+1}/{Config.EPOCHS}")
    loop = tqdm(dataloader, total=len(dataloader), leave=True)
    for batch in loop:
        ids, mask, targets = batch['ids'].to(Config.DEVICE), batch['mask'].to(Config.DEVICE), batch['targets'].to(Config.DEVICE)
        
        optimizer.zero_grad()
        outputs = model(ids, attention_mask=mask).logits
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        fin_loss += loss.item()
        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())
        
    print(f"Epoch {epoch+1} Average Loss: {fin_loss/len(dataloader):.4f}")

def validate(model, dataloader):
    model.eval()
    fin_targets, fin_outputs = [], []
    with torch.no_grad():
        for data in tqdm(dataloader, desc="Validating"):
            ids, mask, targets = data['ids'].to(Config.DEVICE), data['mask'].to(Config.DEVICE), data['targets'].to(Config.DEVICE)
            outputs = model(ids, attention_mask=mask).logits
            fin_targets.extend(targets.cpu().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().numpy().tolist())
    return np.array(fin_outputs), np.array(fin_targets)

# =========================================================================================
# 5. VISUALIZATION MODULE
# =========================================================================================

def plot_comprehensive_metrics(targets, outputs, dataset_name="Unseen Test Data"):
    print(f"\nGenerating Visual Metrics for {dataset_name}...")
    precisions, recalls, rocs = [], [], []
    for i, label in enumerate(Config.LABELS):
        t, p = targets[:, i], outputs[:, i]
        pred = (p >= Config.CUSTOM_THRESHOLDS[label]).astype(int)
        precisions.append(precision_score(t, pred, zero_division=0))
        recalls.append(recall_score(t, pred, zero_division=0))
        try:
            rocs.append(roc_auc_score(t, p))
        except ValueError:
            rocs.append(0)

    x = np.arange(len(Config.LABELS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, precisions, width, label='Precision', color='#2ca02c')
    ax.bar(x, recalls, width, label='Recall', color='#ff7f0e')
    ax.bar(x + width, rocs, width, label='ROC-AUC', color='#1f77b4')
    ax.set_ylabel('Scores (0.0 to 1.0)')
    ax.set_title(f'Per-Class Performance Metrics ({dataset_name})')
    ax.set_xticks(x)
    ax.set_xticklabels(Config.LABELS, rotation=45)
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.1])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# =========================================================================================
# 6. UNSEEN DATA EVALUATION
# =========================================================================================

def evaluate_unseen_data(model, tokenizer, optimized_threshold):
    print("\n" + "="*50)
    print("PHASE 3: EVALUATING ON UNSEEN TEST DATA")
    print("="*50)
    
    test_texts = pd.read_csv(Config.TEST_DATA_PATH)
    test_labels = pd.read_csv(Config.TEST_LABELS_PATH)
    test_df = pd.merge(test_texts, test_labels, on='id')
    test_df = test_df[test_df['toxic'] != -1].reset_index(drop=True)
    test_df['comment_text'] = test_df['comment_text'].apply(clean_text)
    
    test_dataset = ToxicDataset(test_df, tokenizer, Config.MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=Config.VALID_BATCH_SIZE, shuffle=False, num_workers=2)
    
    print("Running Inference on Unseen Test Data...")
    fin_outputs, fin_targets = validate(model, test_loader)
    
    severe_idx = Config.LABELS.index('severe_toxic')
    severe_targets = fin_targets[:, severe_idx]
    
    
    Config.CUSTOM_THRESHOLDS['severe_toxic'] = optimized_threshold
    severe_preds = (fin_outputs[:, severe_idx] >= optimized_threshold).astype(int)
    
    final_prec = precision_score(severe_targets, severe_preds, zero_division=0)
    final_accuracy = accuracy_score(fin_targets, fin_outputs >= 0.5)
    final_roc = roc_auc_score(fin_targets, fin_outputs, average='micro')
    
    print(f"\n=========================================")
    print(f"  FINAL SYSTEM REPORT (UNSEEN DATA)")
    print(f"=========================================")
    print(f"Global Accuracy:         {final_accuracy:.4f}")
    print(f"Micro ROC-AUC Score:     {final_roc:.4f}")
    print(f"Operational Threshold:   {optimized_threshold:.2f}")
    print(f"Severe Toxic Precision:  {final_prec:.4f} (Constraint: >0.80)")
    print(f"=========================================")
    
    plot_comprehensive_metrics(fin_targets, fin_outputs, "Unseen Test Data")

# =========================================================================================
# 7. MAIN EXECUTION PIPELINE
# =========================================================================================

def run_project():
    if not os.path.exists(Config.DATA_PATH):
        print("ERROR: Dataset missing. Please add Jigsaw dataset to Kaggle environment.")
        return

    print("PHASE 1: DATA PREPARATION")
    df = pd.read_csv(Config.DATA_PATH)
    df['comment_text'] = df['comment_text'].apply(clean_text)
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    
    print(f"\nInitializing {Config.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_NAME, num_labels=6)
    
    
    print("Injecting Custom Multi-Sample Dropout Classification Head...")
    model.classifier = RobertaMultiSampleDropoutHead(
        hidden_size=model.config.hidden_size, 
        num_labels=6, 
        num_samples=5
    )
    model.to(Config.DEVICE)
    
    train_dataset = ToxicDataset(train_df.reset_index(drop=True), tokenizer, Config.MAX_LEN)
    val_dataset = ToxicDataset(val_df.reset_index(drop=True), tokenizer, Config.MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=Config.VALID_BATCH_SIZE, shuffle=False, num_workers=2)
    
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = MultiLabelFocalLoss(alpha=0.25, gamma=2.0) 
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * Config.EPOCHS
    )
    
    for epoch in range(Config.EPOCHS):
        train(epoch, model, train_loader, optimizer, criterion, scheduler)
    
    print("\n" + "="*50)
    print("PHASE 2: DYNAMIC PR-CURVE OPTIMIZATION (VALIDATION)")
    print("="*50)
    outputs, targets = validate(model, val_loader)
    
    severe_idx = Config.LABELS.index('severe_toxic')
    t_severe, p_severe = targets[:, severe_idx], outputs[:, severe_idx]
    
    best_thresh, best_prec = 0.50, 0.0
    print("Scanning probabilities to isolate optimal Auto-Ban threshold...")
    for thresh in np.arange(0.50, 0.95, 0.01):
        preds = (p_severe >= thresh).astype(int)
        if preds.sum() == 0: break 
        prec = precision_score(t_severe, preds, zero_division=0)
        
        if prec >= 0.82: 
            best_thresh, best_prec = thresh, prec
            break 
        elif prec > best_prec:
            best_prec, best_thresh = prec, thresh
            
    print(f"Optimization Complete. Locked Threshold: {best_thresh:.2f} (Val Precision: {best_prec:.4f})")
    
    
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(Config.SAVE_DIR, "pytorch_model.bin"))
    tokenizer.save_pretrained(Config.SAVE_DIR)
    
    
    evaluate_unseen_data(model, tokenizer, best_thresh)

if __name__ == "__main__":
    run_project()