# Fashion AI Recommendation System

This project implements an AI-powered fashion recommendation system that combines computer vision and natural language processing to analyze fashion products and provide recommendations based on both visual features and textual reviews.

## 🚀 Project Overview

The Fashion AI Recommendation System consists of three main components:
1. **Vision Model**: Identifies fashion items (shirts, shoes) from images
2. **NLP Model**: Analyzes sentiment from product reviews
3. **Fusion Model**: Combines vision and NLP outputs to generate recommendations

## 📁 Directory Structure

```
fashion_ai/
├── requirements.txt          # Project dependencies
├── .gitignore              # Git ignore rules
├── README.md               # Project documentation
├── scripts/
│   ├── 1_data_check.py     # Data validation
│   ├── 2_create_small_dataset.py  # Dataset preparation
│   ├── 3_image_preprocessing.py   # Image preprocessing
│   ├── 4_check_columns.py  # Column validation
│   ├── 4_text_preprocessing.py    # Text preprocessing
│   ├── 5_train_vision_model.py    # Vision model training
│   ├── 6_train_nlp_model.py       # NLP model training
│   ├── 7_fusion_model.py   # Fusion model training
│   ├── 8_fastapi.py        # API server
│   ├── 9_streamlit.py      # Web interface
│   └── evaluate_models.py  # Model evaluation script
└── models/                 # Model artifacts (not included in repo)
```

## 🛠️ Setup Instructions

1. Clone the repository:
```bash
git clone <repository-url>
cd fashion_ai
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🏗️ Training the Models

1. Prepare your dataset:
```bash
python scripts/2_create_small_dataset.py
```

2. Train the vision model:
```bash
python scripts/5_train_vision_model.py
```

3. Train the NLP model:
```bash
python scripts/6_train_nlp_model.py
```

4. Train the fusion model:
```bash
python scripts/7_fusion_model.py
```

## 📊 Model Evaluation and Accuracy Checking

After training your models, you can evaluate their accuracy using the evaluation script:

```bash
# Evaluate all models
python scripts/evaluate_models.py

# Evaluate specific model
python scripts/evaluate_models.py --model vision    # Only vision model
python scripts/evaluate_models.py --model nlp       # Only NLP model
python scripts/evaluate_models.py --model fusion    # Only fusion model
```

The evaluation script will provide detailed metrics including:
- **Accuracy**: Percentage of correct predictions
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: Ratio of correctly predicted positive observations
- **Recall**: Ratio of correctly predicted positive observations to all actual positives
- **Confusion Matrix**: Detailed breakdown of predictions

## 🌐 Running the Application

1. Start the API server:
```bash
python scripts/8_fastapi.py
```

2. In a new terminal, start the Streamlit interface:
```bash
streamlit run scripts/9_streamlit.py
```

## 🤖 Model Architecture

- **Vision Model**: Custom CNN with convolutional layers for image classification
- **NLP Model**: Transformer-based model for sentiment analysis
- **Fusion Model**: Neural network that combines vision and NLP outputs

## 📝 Notes

- The trained model files (.pth) and datasets are excluded from the repository due to size constraints
- To run the full application, you need to train the models using your own dataset
- The project expects a specific data structure with `data/raw/` containing the original dataset

## 🚀 Deployment

The application can be deployed by:
1. Training all models locally
2. Deploying the FastAPI backend to a cloud service
3. Hosting the Streamlit frontend separately or integrating with the API

## 📄 License

[Add your license information here]