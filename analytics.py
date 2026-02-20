# =========================================================================================
# ADVANCED ANALYTICS
# =========================================================================================

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, hamming_loss, roc_curve, auc, confusion_matrix
from tqdm import tqdm

def generate_final_charts():
    print("Loading Saved Model from Disk...")
    
    
    tokenizer = AutoTokenizer.from_pretrained(Config.SAVE_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_NAME, num_labels=6)
    
    
    model.classifier = RobertaMultiSampleDropoutHead(
        hidden_size=model.config.hidden_size, 
        num_labels=6, 
        num_samples=5
    )
    
    model.load_state_dict(torch.load(os.path.join(Config.SAVE_DIR, "pytorch_model.bin")))
    model.to(Config.DEVICE)
    model.eval()

    
    print("Loading Unseen Test Data...")
    test_texts = pd.read_csv(Config.TEST_DATA_PATH)
    test_labels = pd.read_csv(Config.TEST_LABELS_PATH)
    test_df = pd.merge(test_texts, test_labels, on='id')
    test_df = test_df[test_df['toxic'] != -1].reset_index(drop=True)
    test_df['comment_text'] = test_df['comment_text'].apply(clean_text)
    
    test_dataset = ToxicDataset(test_df, tokenizer, Config.MAX_LEN)
    test_loader = DataLoader(test_dataset, batch_size=Config.VALID_BATCH_SIZE, shuffle=False, num_workers=2)

    
    print("Running Inference...")
    fin_targets = []
    fin_outputs = []
    
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing"):
            ids = data['ids'].to(Config.DEVICE)
            mask = data['mask'].to(Config.DEVICE)
            targets = data['targets'].to(Config.DEVICE)
            
            outputs = model(ids, attention_mask=mask).logits
            fin_targets.extend(targets.cpu().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().numpy().tolist())
            
    fin_outputs = np.array(fin_outputs)
    fin_targets = np.array(fin_targets)

    
    print("\n" + "="*60)
    print(" 🎓 POSTGRAD ADVANCED METRICS REPORT")
    print("="*60)
    
    
    preds = np.zeros_like(fin_outputs)
    for i, label in enumerate(Config.LABELS):
        preds[:, i] = (fin_outputs[:, i] >= Config.CUSTOM_THRESHOLDS[label]).astype(int)

    
    h_loss = hamming_loss(fin_targets, preds)
    print(f"\n1. Hamming Loss: {h_loss:.4f}")
    print("\n2. Detailed Classification Report (F1-Scores):")
    print("-" * 60)
    print(classification_report(fin_targets, preds, target_names=Config.LABELS, zero_division=0))

    
    plt.figure(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for i, label in enumerate(Config.LABELS):
        fpr, tpr, _ = roc_curve(fin_targets[:, i], fin_outputs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{label} (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guessing (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curves by Class', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    for i, label in enumerate(Config.LABELS):
        cm = confusion_matrix(fin_targets[:, i], preds[:, i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False, annot_kws={"size": 14})
        axes[i].set_title(f'Confusion Matrix: {label}', fontsize=12, pad=10)
        axes[i].set_ylabel('Actual', fontsize=10)
        axes[i].set_xlabel('Predicted', fontsize=10)
        axes[i].set_xticklabels(['Negative', 'Positive'])
        axes[i].set_yticklabels(['Negative', 'Positive'])
    plt.suptitle('Per-Class Confusion Matrices (Test Data)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


generate_final_charts()