import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

print("=== TEXT PREPROCESSING ===")

# Download NLTK data
try:
    nltk.download('stopwords')
    nltk.download('wordnet')
    print("✅ NLTK data downloaded")
except:
    print("⚠️ NLTK download issue, but continuing...")

# 1. Load balanced reviews
reviews_df = pd.read_csv("data/processed/small_reviews.csv")
print(f"✅ Loaded {len(reviews_df)} reviews")
print("Reviews distribution:")
print(reviews_df['sentiment'].value_counts())

# 2. Simple text cleaning
def clean_text(text):
    if pd.isna(text):
        return ""
    
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

# 3. Clean the reviews - 'text' column use karein
print("\nCleaning text data...")
reviews_df['cleaned_review'] = reviews_df['text'].apply(clean_text)

# Show samples
print("\nSample original vs cleaned reviews:")
for i in range(2):
    print(f"Original: {reviews_df['text'].iloc[i][:80]}...")
    print(f"Cleaned:  {reviews_df['cleaned_review'].iloc[i][:80]}...")
    print()

# 4. Prepare labels
sentiment_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
reviews_df['sentiment_label'] = reviews_df['sentiment'].map(sentiment_mapping)

print("✅ Sentiment mapping done:")
print(sentiment_mapping)

# 5. Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    reviews_df['cleaned_review'].tolist(),
    reviews_df['sentiment_label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=reviews_df['sentiment_label']
)

print(f"\n✅ Data split completed:")
print(f"Training samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")

print("\n=== TEXT PREPROCESSING COMPLETED ===")
print("Ready for model training!")