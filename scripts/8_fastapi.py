from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from PIL import Image
import numpy as np
import io

print("=== FASTAPI BACKEND ===")

# Models load karein (same as before)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(128 * 28 * 28, 512), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class FusionModel(nn.Module):
    def __init__(self, input_dim=5):
        super(FusionModel, self).__init__()
        self.fusion_net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.fusion_net(x)

# Load models
import os

# Get the absolute path to the project root directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load models with absolute paths
vision_model = SimpleCNN(num_classes=2)
vision_model.load_state_dict(torch.load(os.path.join(project_root, "models", "vision_model.pth")))
vision_model.eval()

nlp_model = AutoModelForSequenceClassification.from_pretrained(os.path.join(project_root, "models", "nlp_model"))
nlp_tokenizer = AutoTokenizer.from_pretrained(os.path.join(project_root, "models", "nlp_model"))
nlp_model.eval()

fusion_model = FusionModel()
fusion_model.load_state_dict(torch.load(os.path.join(project_root, "models", "fusion_model.pth")))
fusion_model.eval()

print("✅ All models loaded for API")

# FastAPI app
app = FastAPI(title="AI Fashion Recommender")

# CORS enable karein
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "Fashion AI Recommendation API is running!"}

@app.post("/predict")
async def predict(
    image: UploadFile = File(...),
    review_text: str = Form(...)
):
    try:
        # Process image
        image_data = await image.read()
        image_pil = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_pil = image_pil.resize((224, 224))
        image_tensor = torch.from_numpy(np.array(image_pil)).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0)
        
        # SIMPLE IMAGE DETECTION - Check if image is too simple
        image_array = np.array(image_pil)
        
        # Calculate image complexity (variance of pixel values)
        complexity = np.var(image_array)
        
        # If image is too simple (plain background, etc.), reject it
        if complexity < 1000:  # Low complexity threshold
            return {
                "success": True,
                "vision_prediction": "Unknown",
                "vision_confidence": 0.3,
                "sentiment": "Unknown", 
                "sentiment_confidence": 0.0,
                "recommendation_score": 0.0,
                "recommendation_status": "NOT A FASHION PRODUCT"
            }
        
        # Vision prediction
        with torch.no_grad():
            vision_output = vision_model(image_tensor)
            vision_probs = torch.softmax(vision_output, dim=1)
            vision_class = torch.argmax(vision_output, dim=1).item()
            vision_confidence = float(vision_probs[0][vision_class])
        
        vision_classes = {0: "Shirt", 1: "Shoe"}
        
        # CHECK: Agar confidence medium hai to "Maybe" return karein
        if vision_confidence < 0.8:  # 80% threshold (strict)
            return {
                "success": True,
                "vision_prediction": f"Maybe {vision_classes[vision_class]}",
                "vision_confidence": vision_confidence,
                "sentiment": "Unknown",
                "sentiment_confidence": 0.0,
                "recommendation_score": 0.0,
                "recommendation_status": "LOW CONFIDENCE - MAY NOT BE FASHION"
            }
        
        # NLP prediction (sirf agar vision confidence high hai)
        with torch.no_grad():
            text_encoding = nlp_tokenizer(review_text, return_tensors='pt', truncation=True, padding=True, max_length=128)
            nlp_output = nlp_model(**text_encoding)
            nlp_probs = torch.softmax(nlp_output.logits, dim=1)
            nlp_class = torch.argmax(nlp_output.logits, dim=1).item()
            nlp_confidence = float(nlp_probs[0][nlp_class])
        
        nlp_classes = {0: "Negative", 1: "Neutral", 2: "Positive"}
        
        # Fusion prediction
        combined_features = torch.cat([vision_probs.squeeze(), nlp_probs.squeeze()])
        with torch.no_grad():
            recommendation_score = fusion_model(combined_features.unsqueeze(0))
        
        recommendation_status = "HIGHLY RECOMMENDED" if recommendation_score > 0.7 else "MODERATELY RECOMMENDED" if recommendation_score > 0.4 else "NOT RECOMMENDED"
        
        return {
            "success": True,
            "vision_prediction": vision_classes[vision_class],
            "vision_confidence": vision_confidence,
            "sentiment": nlp_classes[nlp_class],
            "sentiment_confidence": nlp_confidence,
            "recommendation_score": float(recommendation_score.item()),
            "recommendation_status": recommendation_status
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI server...")
    print("📡 API will be available at: http://localhost:8000")
    print("📋 API Documentation: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)