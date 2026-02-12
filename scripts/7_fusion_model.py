import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import time
import os
from PIL import Image
from transformers import AutoModelForSequenceClassification, AutoTokenizer

print("=== FUSION MODEL TRAINING ===")

# 1. Simple CNN Model (Same as vision training)
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

# 2. Load trained models
print("Loading pre-trained models...")

# Vision model
vision_model = SimpleCNN(num_classes=2)
vision_model.load_state_dict(torch.load("models/vision_model.pth"))
vision_model.eval()
print("Vision model loaded")

# NLP model
nlp_model = AutoModelForSequenceClassification.from_pretrained("models/nlp_model/")
nlp_tokenizer = AutoTokenizer.from_pretrained("models/nlp_model/")
nlp_model.eval()
print("NLP model loaded")

# 3. Simple image transform
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
    return image_tensor.unsqueeze(0)  # Add batch dimension

# 4. Simple text cleaning
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    return text

# 5. Create fusion dataset
class FusionDataset(Dataset):
    def __init__(self, styles_csv, reviews_csv, image_dir, sample_size=500):
        self.styles_df = pd.read_csv(styles_csv)
        self.reviews_df = pd.read_csv(reviews_csv)
        self.image_dir = image_dir
        
        # Use smaller sample for fast training
        self.sample_size = min(sample_size, len(self.styles_df), len(self.reviews_df))
        
    def __len__(self):
        return self.sample_size
    
    def __getitem__(self, idx):
        # Get image data
        style_row = self.styles_df.iloc[idx]
        image_id = style_row['id']
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        
        # Get vision prediction
        with torch.no_grad():
            image_tensor = preprocess_image(image_path)
            vision_output = vision_model(image_tensor)
            vision_probs = torch.softmax(vision_output, dim=1)
        
        # Get text data
        review_row = self.reviews_df.iloc[idx]
        review_text = review_row['text']
        cleaned_text = clean_text(review_text)
        
        # Get NLP prediction
        with torch.no_grad():
            text_encoding = nlp_tokenizer(cleaned_text, return_tensors='pt', truncation=True, padding=True, max_length=128)
            nlp_output = nlp_model(**text_encoding)
            nlp_probs = torch.softmax(nlp_output.logits, dim=1)
        
        # Combine features (vision probs + nlp probs)
        combined_features = torch.cat([vision_probs.squeeze(), nlp_probs.squeeze()])
        
        # Target: Use rating for recommendation (rating >= 4 = good recommendation)
        rating = review_row['rating']
        target = 1 if rating >= 4 else 0
        
        return combined_features, torch.tensor(target, dtype=torch.float)

# 6. Simple Fusion Model
class FusionModel(nn.Module):
    def __init__(self, input_dim=5):  # 2 (vision) + 3 (nlp) = 5
        super(FusionModel, self).__init__()
        self.fusion_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fusion_net(x)

# 7. Create fusion dataset
print("Creating fusion dataset...")
fusion_dataset = FusionDataset(
    styles_csv="data/processed/small_styles.csv",
    reviews_csv="data/processed/small_reviews.csv", 
    image_dir="data/processed/small_images/",
    sample_size=400  # Small for fast training
)

# Split data
train_size = int(0.8 * len(fusion_dataset))
val_size = len(fusion_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(fusion_dataset, [train_size, val_size])

train_loader = DataLoader(train_subset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=16, shuffle=False)

print(f"Training samples: {len(train_subset)}")
print(f"Validation samples: {len(val_subset)}")

# 8. Initialize fusion model
fusion_model = FusionModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(fusion_model.parameters(), lr=0.001)

print("Fusion model created")

# 9. Training function
def train_fusion_model(model, train_loader, val_loader, epochs=3):
    for epoch in range(epochs):
        print(f"\nFusion Epoch {epoch+1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0
        for batch_idx, (features, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            
            outputs = model(features)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for features, targets in val_loader:
                outputs = model(features)
                preds = (outputs.squeeze() > 0.5).float()
                all_preds.extend(preds.numpy())
                all_targets.extend(targets.numpy())
        
        accuracy = accuracy_score(all_targets, all_preds)
        
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Accuracy: {accuracy:.4f}")
    
    return model

# 10. Train fusion model
print("\nStarting Fusion Training...")
start_time = time.time()

fusion_model = train_fusion_model(fusion_model, train_loader, val_loader, epochs=3)

end_time = time.time()
print(f"\nFusion training completed in {(end_time - start_time)/60:.2f} minutes")

# 11. Save fusion model
torch.save(fusion_model.state_dict(), "models/fusion_model.pth")
print("Fusion model saved as 'models/fusion_model.pth'")

print("\n=== FUSION MODEL TRAINING COMPLETED ===")
print("Ready for deployment!")