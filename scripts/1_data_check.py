import pandas as pd
import os

print("=== DATA CHECK ===")

# Pehle check karein files exist karti hain ya nahi
print("1. Checking if files exist...")
print(f"styles.csv exists: {os.path.exists('data/raw/styles.csv')}")
print(f"reviews.csv exists: {os.path.exists('data/raw/reviews.csv')}")
print(f"images folder exists: {os.path.exists('data/raw/images/')}")

# CSV files ke first few lines check karein
print("\n2. Checking CSV file structure...")
try:
    # Pehle 5 lines read karein
    with open("data/raw/styles.csv", 'r') as f:
        lines = [f.readline() for _ in range(5)]
    print("First 5 lines of styles.csv:")
    for i, line in enumerate(lines):
        print(f"Line {i+1}: {line.strip()}")
except Exception as e:
    print(f"Error reading styles.csv: {e}")

# Images count check karein
print("\n3. Checking images...")
try:
    images = os.listdir("data/raw/images/")
    print(f"Total images: {len(images)}")
    print(f"First 5 image names: {images[:5]}")
except Exception as e:
    print(f"Error reading images: {e}")