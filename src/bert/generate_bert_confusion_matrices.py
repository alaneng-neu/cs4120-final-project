import os
import joblib
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, 
    DistilBertForSequenceClassification,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AutoConfig
)

# Constants
RANDOM_SEED = 42
BATCH_SIZE = 16
MAX_LEN = 128
MODEL_NAME = 'distilbert-base-uncased'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Dataset Classes ---

class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class PhishingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- Helper Functions ---

def load_spam_data():
    data_path = os.path.join(os.path.dirname(__file__), "../../data/enron_spam_data.csv")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['Spam/Ham'])
    
    # Replicate the downsampling logic from training if it was used
    # train_spam_bert.py: if len(df) > 20000: df = df.sample(n=20000, random_state=RANDOM_SEED)
    if len(df) > 20000:
        print(f"Downsampling spam dataset from {len(df)} to 20000 samples (matching training logic)...")
        df = df.sample(n=20000, random_state=RANDOM_SEED)
    
    X = (df['Subject'].fillna("") + " " + df['Message'].fillna("")).astype(str)
    y_raw = df['Spam/Ham'].astype(str)
    return X, y_raw

def load_phishing_data():
    data_path = os.path.join(os.path.dirname(__file__), "../../data/phishing_legit_dataset_cleaned.csv")
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['text', 'phishing_type'])
    X = df['text'].fillna("").astype(str)
    y_raw = df['phishing_type'].astype(str)
    return X, y_raw

def eval_model(model, data_loader, device):
    model = model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_labels), np.array(all_preds)

def plot_combined_cm(y_true_train, y_pred_train, y_true_test, y_pred_test, class_names, title, filename):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Train Matrix
    cm_train = confusion_matrix(y_true_train, y_pred_train)
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f"{title} (Training Set)")
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Test Matrix
    cm_test = confusion_matrix(y_true_test, y_pred_test)
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f"{title} (Test Set)")
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved combined confusion matrix plot to {filename}")
    plt.close()
    
    print(f"\n--- {title} Training Set Report ---")
    print(classification_report(y_true_train, y_pred_train, target_names=class_names, digits=4))
    
    print(f"\n--- {title} Test Set Report ---")
    print(classification_report(y_true_test, y_pred_test, target_names=class_names, digits=4))


def main():
    models_dir = os.path.join(os.path.dirname(__file__), "../../models")
    
    # ---------------------------------------------------------
    # 1. SPAM BERT MODEL
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("Evaluating SPAM BERT Model...")
    print("="*50)
    
    spam_model_path = os.path.join(models_dir, "spam_bert_model")
    spam_encoder_path = os.path.join(models_dir, "spam_bert_encoder.joblib")
    
    if os.path.exists(spam_model_path) and os.path.exists(spam_encoder_path):
        # Load Data
        X_spam, y_spam_raw = load_spam_data()
        
        # Load Encoder
        spam_encoder = joblib.load(spam_encoder_path)
        y_spam = spam_encoder.transform(y_spam_raw)
        
        # Split (to get the same test set)
        X_train_spam, X_test_spam, y_train_spam, y_test_spam = train_test_split(
            X_spam, y_spam, test_size=0.2, random_state=RANDOM_SEED, stratify=y_spam
        )
        print(f"Train size: {len(X_train_spam)}")
        print(f"Test size: {len(X_test_spam)}")
        
        # Load Config to check model type
        config = AutoConfig.from_pretrained(spam_model_path)
        model_type = config.model_type
        print(f"Detected Spam Model Type: {model_type}")

        if model_type == 'roberta':
            tokenizer = RobertaTokenizer.from_pretrained(spam_model_path)
            model = RobertaForSequenceClassification.from_pretrained(spam_model_path)
        else:
            tokenizer = DistilBertTokenizer.from_pretrained(spam_model_path)
            model = DistilBertForSequenceClassification.from_pretrained(spam_model_path)
            
        model.to(device)
        
        # --- Evaluate Training Set ---
        train_dataset = SpamDataset(
            texts=X_train_spam.to_numpy(),
            labels=y_train_spam,
            tokenizer=tokenizer,
            max_len=MAX_LEN
        )
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
        print("Evaluating on Training Set (this may take a moment)...")
        y_true_train, y_pred_train = eval_model(model, train_loader, device)

        # --- Evaluate Test Set ---
        test_dataset = SpamDataset(
            texts=X_test_spam.to_numpy(),
            labels=y_test_spam,
            tokenizer=tokenizer,
            max_len=MAX_LEN
        )
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        print("Evaluating on Test Set...")
        y_true_test, y_pred_test = eval_model(model, test_loader, device)
        
        # Plot Combined
        plot_combined_cm(
            y_true_train, y_pred_train,
            y_true_test, y_pred_test,
            class_names=spam_encoder.classes_, 
            title="Spam Detection BERT", 
            filename="confusion_matrix_spam_bert_combined.png"
        )
    else:
        print("Spam BERT model or encoder not found. Skipping.")

    # ---------------------------------------------------------
    # 2. PHISHING BERT MODEL
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("Evaluating PHISHING BERT Model...")
    print("="*50)
    
    phish_model_path = os.path.join(models_dir, "phishing_bert_model")
    phish_encoder_path = os.path.join(models_dir, "phishing_bert_encoder.joblib")
    
    if os.path.exists(phish_model_path) and os.path.exists(phish_encoder_path):
        # Load Data
        X_phish, y_phish_raw = load_phishing_data()
        
        # Load Encoder
        phish_encoder = joblib.load(phish_encoder_path)
        y_phish = phish_encoder.transform(y_phish_raw)
        
        # Split (to get the same test set)
        _, X_test_phish, _, y_test_phish = train_test_split(
            X_phish, y_phish, test_size=0.2, random_state=RANDOM_SEED, stratify=y_phish
        )
        print(f"Test size: {len(X_test_phish)}")
        
        # Load Tokenizer & Model
        tokenizer = DistilBertTokenizer.from_pretrained(phish_model_path)
        model = DistilBertForSequenceClassification.from_pretrained(phish_model_path)
        model.to(device)
        
        # Dataset & Loader
        test_dataset = PhishingDataset(
            texts=X_test_phish.to_numpy(),
            labels=y_test_phish,
            tokenizer=tokenizer,
            max_len=MAX_LEN
        )
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        # Evaluate
        y_true, y_pred = eval_model(model, test_loader, device)
        
        # Plot (Test only for Phishing as requested, or switch to combined if needed later)
        # Using the old plot_cm function here, we need to redefine it or update this call
        # Since we replaced plot_cm with plot_combined_cm, we should add back a simple plot function 
        # or adapt this part. Let's add a simple plot function back.
        
        # Re-adding simple plot function logic inline or separate? 
        # Easier to just use the combined logic but pass None for train if we didn't calculate it?
        # Or just calculate train for Phishing too? The user only asked for Spam.
        # Let's restore a simple plot_cm for Phishing to avoid breaking it.
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=phish_encoder.classes_, yticklabels=phish_encoder.classes_)
        plt.title("Phishing Type BERT Confusion Matrix")
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig("confusion_matrix_phishing_bert.png")
        plt.close()
        
        print(f"Saved confusion matrix plot to confusion_matrix_phishing_bert.png")
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=phish_encoder.classes_, digits=4))
    else:
        print("Phishing BERT model or encoder not found. Skipping.")

if __name__ == '__main__':
    main()

