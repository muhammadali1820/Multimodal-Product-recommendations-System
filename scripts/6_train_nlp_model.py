import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
import time
import re

print("=== NLP MODEL TRAINING ===")

# 1. Load and clean reviews
reviews_df = pd.read_csv("data/processed/small_reviews.csv")
print(f"Loaded {len(reviews_df)} reviews")

# Simple text cleaning function
def clean_text(text):
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

# Clean the text
print("Cleaning text data...")
reviews_df['cleaned_text'] = reviews_df['text'].apply(clean_text)

# Remove empty reviews
reviews_df = reviews_df[reviews_df['cleaned_text'].str.len() > 10]
print(f"Using {len(reviews_df)} valid reviews after cleaning")

# 2. Prepare labels
sentiment_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
reviews_df['sentiment_label'] = reviews_df['sentiment'].map(sentiment_mapping)

print("Label distribution:")
print(reviews_df['sentiment'].value_counts())

# 3. Split data
from sklearn.model_selection import train_test_split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    reviews_df['cleaned_text'].tolist(),
    reviews_df['sentiment_label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=reviews_df['sentiment_label']
)

print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")

# 4. Initialize tokenizer (DistilBERT - Fast and efficient)
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize data
print("Tokenizing data...")

# Create Dataset class with proper tokenization
class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def __len__(self):
        return len(self.texts)

train_dataset = ReviewDataset(train_texts, train_labels, tokenizer)
val_dataset = ReviewDataset(val_texts, val_labels, tokenizer)

# 5. Load pretrained model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=3  # 3 classes: negative, neutral, positive
)

print("DistilBERT model loaded")

# 6. SIMPLE TRAINING LOOP (No Trainer class issues)
def simple_train_model(model, train_dataset, val_dataset, epochs=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"   Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        print(f"   Train Loss: {avg_loss:.4f}")
        print(f"   Val Accuracy: {accuracy:.4f}")
        print(f"   Val F1-Score: {f1:.4f}")
    
    return model

# 7. Train the model
print("\nStarting NLP Training...")
start_time = time.time()

model = simple_train_model(model, train_dataset, val_dataset, epochs=3)

end_time = time.time()
print(f"\nNLP Training completed in {(end_time - start_time)/60:.2f} minutes")

# 8. Save model
model.save_pretrained("models/nlp_model")
tokenizer.save_pretrained("models/nlp_model")
print("NLP model saved to 'models/nlp_model/'")

print("\n=== NLP MODEL TRAINING COMPLETED ===")