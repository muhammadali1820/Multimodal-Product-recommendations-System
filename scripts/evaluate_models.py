import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from PIL import Image
import os
import time
import argparse

print("=== MODEL EVALUATION SCRIPT ===")

# 1. Vision Model Definition (same as training)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 2. Fusion Model Definition (same as training)
class FusionModel(nn.Module):
    def __init__(self, input_dim=5):
        super(FusionModel, self).__init__()
        self.fusion_net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.fusion_net(x)

# 3. Vision Dataset Class
class VisionDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        # Filter only images that exist
        self.valid_indices = []
        for idx in range(len(self.data)):
            image_id = self.data.iloc[idx]['id']
            image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
            if os.path.exists(image_path):
                self.valid_indices.append(idx)

        print(f"Found {len(self.valid_indices)} valid images out of {len(self.data)}")

        # Create label mapping
        self.classes = ['Shirts', 'Casual Shoes']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        row = self.data.iloc[actual_idx]
        image_id = row['id']
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")

        # Load image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # Get label
        label = self.class_to_idx[row['articleType']]

        return image_tensor, label

# 4. NLP Dataset Class
class NLPDataset(Dataset):
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

# 5. Fusion Dataset Class
class FusionDataset(Dataset):
    def __init__(self, styles_csv, reviews_csv, image_dir, sample_size=500):
        self.styles_df = pd.read_csv(styles_csv)
        self.reviews_df = pd.read_csv(reviews_csv)
        self.image_dir = image_dir

        # Use smaller sample for fast evaluation
        self.sample_size = min(sample_size, len(self.styles_df), len(self.reviews_df))

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        # Get image data
        style_row = self.styles_df.iloc[idx]
        image_id = style_row['id']
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")

        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = image.resize((224, 224))
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        # Get vision prediction
        with torch.no_grad():
            vision_output = self.vision_model(image_tensor)
            vision_probs = torch.softmax(vision_output, dim=1)

        # Get text data
        review_row = self.reviews_df.iloc[idx]
        review_text = review_row['text']

        # Get NLP prediction
        with torch.no_grad():
            text_encoding = self.nlp_tokenizer(review_text, return_tensors='pt', truncation=True, padding=True, max_length=128)
            nlp_output = self.nlp_model(**text_encoding)
            nlp_probs = torch.softmax(nlp_output.logits, dim=1)

        # Combine features (vision probs + nlp probs)
        combined_features = torch.cat([vision_probs.squeeze(), nlp_probs.squeeze()])

        # Target: Use rating for recommendation (rating >= 4 = good recommendation)
        rating = review_row['rating']
        target = 1 if rating >= 4 else 0

        return combined_features, torch.tensor(target, dtype=torch.float)

# 6. Evaluate Vision Model
def evaluate_vision_model():
    print("\nEvaluating Vision Model...")
    
    # Load model
    vision_model = SimpleCNN(num_classes=2)
    vision_model.load_state_dict(torch.load("models/vision_model.pth", map_location=torch.device('cpu')))
    vision_model.eval()
    print("Vision model loaded")
    
    # Create dataset
    dataset = VisionDataset(
        csv_file="data/processed/small_styles.csv",
        image_dir="data/processed/small_images/"
    )
    
    # Split data (80% train, 20% test)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_subset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
    
    # Evaluation
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = vision_model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision, recall, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    
    print(f"Vision Model Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1-Score (weighted): {f1:.4f}")
    print(f"   Precision (macro): {precision:.4f}")
    print(f"   Recall (macro): {recall:.4f}")
    print(f"   F1-Score (macro): {f1_macro:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"   Confusion Matrix:\n{cm}")
    
    return accuracy, f1

# 7. Evaluate NLP Model
def evaluate_nlp_model():
    print("\nEvaluating NLP Model...")
    
    # Load model and tokenizer
    nlp_model = AutoModelForSequenceClassification.from_pretrained("models/nlp_model/")
    nlp_tokenizer = AutoTokenizer.from_pretrained("models/nlp_model/")
    nlp_model.eval()
    print("NLP model loaded")
    
    # Load and prepare data
    reviews_df = pd.read_csv("data/processed/small_reviews.csv")
    
    # Prepare labels
    sentiment_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
    reviews_df['sentiment_label'] = reviews_df['sentiment'].map(sentiment_mapping)
    
    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        reviews_df['text'].tolist(),
        reviews_df['sentiment_label'].tolist(),
        test_size=0.2,
        random_state=42,
        stratify=reviews_df['sentiment_label']
    )
    
    # Create test dataset
    test_dataset = NLPDataset(test_texts, test_labels, nlp_tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Evaluation
    all_preds = []
    all_labels = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nlp_model.to(device)
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = nlp_model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision, recall, f1_macro, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')
    
    print(f"NLP Model Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1-Score (weighted): {f1:.4f}")
    print(f"   Precision (macro): {precision:.4f}")
    print(f"   Recall (macro): {recall:.4f}")
    print(f"   F1-Score (macro): {f1_macro:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print(f"   Confusion Matrix:\n{cm}")
    
    return accuracy, f1

# 8. Evaluate Fusion Model
def evaluate_fusion_model():
    print("\nEvaluating Fusion Model...")
    
    # Load models
    vision_model = SimpleCNN(num_classes=2)
    vision_model.load_state_dict(torch.load("models/vision_model.pth", map_location=torch.device('cpu')))
    vision_model.eval()
    
    nlp_model = AutoModelForSequenceClassification.from_pretrained("models/nlp_model/")
    nlp_tokenizer = AutoTokenizer.from_pretrained("models/nlp_model/")
    nlp_model.eval()
    
    fusion_model = FusionModel()
    fusion_model.load_state_dict(torch.load("models/fusion_model.pth", map_location=torch.device('cpu')))
    fusion_model.eval()
    
    print("All models loaded for fusion evaluation")
    
    # Create fusion dataset with references to models
    class FusionEvalDataset(Dataset):
        def __init__(self, styles_csv, reviews_csv, image_dir, sample_size=500):
            self.styles_df = pd.read_csv(styles_csv)
            self.reviews_df = pd.read_csv(reviews_csv)
            self.image_dir = image_dir
            self.vision_model = vision_model
            self.nlp_model = nlp_model
            self.nlp_tokenizer = nlp_tokenizer

            # Use smaller sample for fast evaluation
            self.sample_size = min(sample_size, len(self.styles_df), len(self.reviews_df))

        def __len__(self):
            return self.sample_size

        def __getitem__(self, idx):
            # Get image data
            style_row = self.styles_df.iloc[idx]
            image_id = style_row['id']
            image_path = os.path.join(self.image_dir, f"{image_id}.jpg")

            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image = image.resize((224, 224))
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

            # Get vision prediction
            with torch.no_grad():
                vision_output = self.vision_model(image_tensor)
                vision_probs = torch.softmax(vision_output, dim=1)

            # Get text data
            review_row = self.reviews_df.iloc[idx]
            review_text = review_row['text']

            # Get NLP prediction
            with torch.no_grad():
                text_encoding = self.nlp_tokenizer(review_text, return_tensors='pt', truncation=True, padding=True, max_length=128)
                nlp_output = self.nlp_model(**text_encoding)
                nlp_probs = torch.softmax(nlp_output.logits, dim=1)

            # Combine features (vision probs + nlp probs)
            combined_features = torch.cat([vision_probs.squeeze(), nlp_probs.squeeze()])

            # Target: Use rating for recommendation (rating >= 4 = good recommendation)
            rating = review_row['rating']
            target = 1 if rating >= 4 else 0

            return combined_features, torch.tensor(target, dtype=torch.float)
    
    # Create dataset
    fusion_dataset = FusionEvalDataset(
        styles_csv="data/processed/small_styles.csv",
        reviews_csv="data/processed/small_reviews.csv",
        image_dir="data/processed/small_images/",
        sample_size=400
    )
    
    # Split data
    train_size = int(0.8 * len(fusion_dataset))
    test_size = len(fusion_dataset) - train_size
    _, test_subset = torch.utils.data.random_split(fusion_dataset, [train_size, test_size])
    
    test_loader = DataLoader(test_subset, batch_size=16, shuffle=False)
    
    # Evaluation
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in test_loader:
            outputs = fusion_model(features)
            preds = (outputs.squeeze() > 0.5).float()
            all_preds.extend(preds.numpy())
            all_targets.extend(targets.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='weighted')
    precision, recall, f1_macro, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')
    
    print(f"Fusion Model Results:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1-Score (weighted): {f1:.4f}")
    print(f"   Precision (macro): {precision:.4f}")
    print(f"   Recall (macro): {recall:.4f}")
    print(f"   F1-Score (macro): {f1_macro:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(all_targets, all_preds)
    print(f"   Confusion Matrix:\n{cm}")
    
    return accuracy, f1

def main():
    print("Starting Model Evaluation...")
    
    parser = argparse.ArgumentParser(description='Evaluate Fashion AI Models')
    parser.add_argument('--model', type=str, choices=['vision', 'nlp', 'fusion', 'all'], default='all',
                        help='Which model to evaluate (default: all)')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    if args.model in ['vision', 'all']:
        vision_acc, vision_f1 = evaluate_vision_model()
    
    if args.model in ['nlp', 'all']:
        nlp_acc, nlp_f1 = evaluate_nlp_model()
    
    if args.model in ['fusion', 'all']:
        fusion_acc, fusion_f1 = evaluate_fusion_model()
    
    end_time = time.time()
    print(f"\nEvaluation completed in {(end_time - start_time):.2f} seconds")

if __name__ == "__main__":
    main()