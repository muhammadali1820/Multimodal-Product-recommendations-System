import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import pandas as pd
import os
import matplotlib.pyplot as plt

print("=== IMAGE PREPROCESSING ===")

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

print("✅ Image transformations defined")

# 2. Create Dataset Class
class FashionDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        
        # Create label mapping
        self.classes = self.data['articleType'].unique()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
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

print("✅ Dataset created successfully")
print(f"Total images: {len(train_dataset)}")
print(f"Classes: {train_dataset.classes}")
print(f"Class mapping: {train_dataset.class_to_idx}")

# 4. Create data loader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print("✅ DataLoader created")
print(f"Batch size: 32")
print(f"Number of batches: {len(train_loader)}")

# 5. Test one batch
print("\nTesting one batch...")
for images, labels in train_loader:
    print(f"Batch images shape: {images.shape}")  # [32, 3, 224, 224]
    print(f"Batch labels shape: {labels.shape}")  # [32]
    print(f"Sample labels: {labels[:5]}")
    break

print("=== IMAGE PREPROCESSING COMPLETED ===")