import pandas as pd
import os
import shutil

print("=== CREATING BALANCED SMALL DATASET ===")

# 1. CSV file ko properly read karein with error handling
try:
    styles = pd.read_csv("data/raw/styles.csv", on_bad_lines='skip')
    print(f"Styles CSV loaded successfully: {len(styles)} rows")
except:
    styles = pd.read_csv("data/raw/styles.csv", error_bad_lines=False)
    print(f"Styles CSV loaded with error handling: {len(styles)} rows")

# Available categories check karein
print("\nAvailable categories in dataset:")
print(styles['articleType'].value_counts().head(10))

# Categories select karein
selected_categories = ['Shirts', 'Casual Shoes']
print(f"\nSelected categories: {selected_categories}")

small_styles = pd.DataFrame()
for category in selected_categories:
    if category in styles['articleType'].values:
        category_data = styles[styles['articleType'] == category].head(750)
        small_styles = pd.concat([small_styles, category_data])
        print(f"Selected {len(category_data)} {category}")
    else:
        print(f"Category '{category}' not found")

if len(small_styles) == 0:
    print("Using alternative categories...")
    alternative_categories = ['Tshirts', 'Shoes', 'Footwear']
    for category in alternative_categories:
        if category in styles['articleType'].values:
            category_data = styles[styles['articleType'] == category].head(750)
            small_styles = pd.concat([small_styles, category_data])
            print(f"Selected {len(category_data)} {category}")

print(f"Total selected: {len(small_styles)} products")

# 2. Images copy karein
small_images_dir = "data/processed/small_images/"
os.makedirs(small_images_dir, exist_ok=True)

copied_count = 0
for index, row in small_styles.iterrows():
    try:
        image_id = row['id']
        image_path = os.path.join("data/raw/images/", f"{image_id}.jpg")
        
        if os.path.exists(image_path):
            shutil.copy2(image_path, small_images_dir)
            copied_count += 1
            
        if copied_count % 100 == 0:
            print(f"Copied {copied_count} images...")
    except:
        pass

print(f"Copied {copied_count} images to {small_images_dir}")

# 3. Small styles CSV save karein
small_styles.to_csv("data/processed/small_styles.csv", index=False)
print("Small styles CSV saved!")

# 4. Balanced reviews banayein
try:
    reviews = pd.read_csv("data/raw/reviews.csv", on_bad_lines='skip')
    
    def get_sentiment(rating):
        if rating >= 4:
            return 'positive'
        elif rating <= 2:
            return 'negative'
        else:
            return 'neutral'

    reviews['sentiment'] = reviews['rating'].apply(get_sentiment)

    sentiments = ['positive', 'negative', 'neutral']
    balanced_reviews = pd.DataFrame()

    for sentiment in sentiments:
        sentiment_data = reviews[reviews['sentiment'] == sentiment].head(667)
        balanced_reviews = pd.concat([balanced_reviews, sentiment_data])

    print(f"\nBalanced reviews distribution:")
    print(balanced_reviews['sentiment'].value_counts())

    balanced_reviews.to_csv("data/processed/small_reviews.csv", index=False)
    print(f"Selected {len(balanced_reviews)} balanced reviews")
    
except Exception as e:
    print(f"Error processing reviews: {e}")

print("=== SMALL DATASET CREATION COMPLETED ===")