import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import time
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import pandas as pd
import os

print("=== PROFESSIONAL VISION MODEL TRAINING ===")

# 1. Define image transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# 2. Dataset Class with ERROR HANDLING
class FashionDataset(Dataset):
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
        
        # Get label
        label = self.class_to_idx[row['articleType']]
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
            
        return image, label

# 3. Create datasets
train_dataset = FashionDataset(
    csv_file="data/processed/small_styles.csv",
    image_dir="data/processed/small_images/", 
    transform=train_transform
)

print(f"Dataset loaded: {len(train_dataset)} valid images")

# Split data (80% train, 20% val)
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

print(f"Training samples: {len(train_subset)}")
print(f"Validation samples: {len(val_subset)}")

# Data loaders
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)

# 4. SIMPLE CNN MODEL (Fast and Reliable)
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

# Initialize model
model = SimpleCNN(num_classes=2)
print("Simple CNN Model created")

# 5. Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Training function
def train_model_simple(model, train_loader, val_loader, epochs=4):
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training
        model.train()
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"   Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Metrics
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        val_accuracies.append(accuracy)
        
        print(f"   Train Loss: {avg_train_loss:.4f}")
        print(f"   Val Accuracy: {accuracy:.4f}")
        print(f"   Val F1-Score: {f1:.4f}")
    
    return train_losses, val_accuracies

# 7. Train the model
print("\nStarting Training...")
start_time = time.time()

train_losses, val_accuracies = train_model_simple(model, train_loader, val_loader, epochs=4)

end_time = time.time()
print(f"\nTraining completed in {(end_time - start_time)/60:.2f} minutes")

# 8. Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/vision_model.pth")
print("Model saved as 'models/vision_model.pth'")

print("\n=== VISION TRAINING COMPLETED ===")