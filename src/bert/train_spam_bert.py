import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from torch.optim import AdamW
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
import joblib
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_utils import evaluate_multiclass_model, plot_confusion_matrix

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LEN = 128  # Max sequence length for BERT
MODEL_NAME = 'distilbert-base-uncased'

def load_data():
    data_path = os.path.join(os.path.dirname(__file__), "../../data/enron_spam_data.csv")
    df = pd.read_csv(data_path)
    print("Dataset shape:", df.shape)
    
    # Drop rows with missing labels
    df = df.dropna(subset=['Spam/Ham'])

    if len(df) > 20000:
        print(f"Downsampling dataset from {len(df)} to 20000 samples for faster training...")
        df = df.sample(n=20000, random_state=RANDOM_SEED)
    
    X = (df['Subject'].fillna("") + " " + df['Message'].fillna("")).astype(str)
    y_raw = df['Spam/Ham'].astype(str)
    
    return X, y_raw

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

def train_epoch(model, data_loader, optimizer, scheduler, device, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0

    for batch in data_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits

        _, preds = torch.max(logits, dim=1)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, device, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0
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

            loss = outputs.loss
            logits = outputs.logits

            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return correct_predictions.double() / n_examples, np.mean(losses), np.array(all_labels), np.array(all_preds)

def main():
    # Load Data
    X, y_raw = load_data()
    
    # Encode Labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    num_classes = len(label_encoder.classes_)
    print(f"Classes ({num_classes}):", list(label_encoder.classes_))
    
    # Split Data

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Tokenizer
    print(f"Loading {MODEL_NAME} tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    
    # Create Datasets
    train_dataset = SpamDataset(
        texts=X_train.to_numpy(),
        labels=y_train,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    test_dataset = SpamDataset(
        texts=X_test.to_numpy(),
        labels=y_test,
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    # Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Model
    print(f"Loading {MODEL_NAME} model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=num_classes
    )
    model = model.to(device)
    
    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # Training Loop
    best_accuracy = 0
    
    print("\nStarting training...")
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("-" * 10)
        
        train_acc, train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            len(train_dataset)
        )
        print(f"Train loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
        
        val_acc, val_loss, _, _ = eval_model(
            model,
            test_loader,
            device,
            len(test_dataset)
        )
        print(f"Val   loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
        print()
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            # Save best model state
            torch.save(model.state_dict(), 'best_spam_bert_model_state.bin')
            
    # Load best model
    if os.path.exists('best_spam_bert_model_state.bin'):
        model.load_state_dict(torch.load('best_spam_bert_model_state.bin'))
        
    # Final Evaluation
    print("\nFinal Evaluation on Test Set:")
    _, _, y_true, y_pred = eval_model(
        model,
        test_loader,
        device,
        len(test_dataset)
    )
    
    # Metrics
    evaluate_multiclass_model(
        y_true=y_true, 
        y_pred=y_pred, 
        dataset_name="Spam BERT (Test Set)"
    )
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
    
    # Save Model
    models_dir = os.path.join(os.path.dirname(__file__), "../../models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Save the model
    model_save_path = os.path.join(models_dir, "spam_bert_model")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"Saved BERT model and tokenizer to {model_save_path}")
    
    encoder_path = os.path.join(models_dir, "spam_bert_encoder.joblib")
    joblib.dump(label_encoder, encoder_path)
    print(f"Saved label encoder to {encoder_path}")
    
    # Clean up temp file
    if os.path.exists('best_spam_bert_model_state.bin'):
        os.remove('best_spam_bert_model_state.bin')

if __name__ == '__main__':
    main()

