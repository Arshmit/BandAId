"""
AI Model for Wound Assessment using MobileNetV3
"""
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_large
from PIL import Image
import io
import os


class WoundModel:
    """Wound assessment model using MobileNetV3"""
    
    def __init__(self):
        """Initialize the model"""
        print("🤖 Loading MobileNetV3 wound assessment model...")
        
        # Path to your model
        model_path = os.path.join(os.path.dirname(__file__), '..', 'MobilenetV3', 'best_surgwound_mobilenetv3.pt')
        
        # Load feature extractor
        self.feature_extractor = mobilenet_v3_large(pretrained=True)
        self.feature_extractor.classifier = nn.Identity()
        self.feature_extractor.eval()
        
        # Define wound classes
        self.wound_classes = {
            'healing_status': {0: 'Not Healed', 1: 'Healed'},
            'edema': {0: 'Non-existent', 1: 'Existent'},
            'erythema': {0: 'Existent', 1: 'Non-existent'},
            'exudate_type': {
                0: 'Non-existent',
                1: 'Sanguineous',
                2: 'Serous',
                3: 'Purulent',
                4: 'Seropurulent'
            },
            'infection_risk': {0: 'Low', 1: 'Medium', 2: 'High'},
            'urgency': {
                0: 'Home Care (Green)',
                1: 'Clinic Visit (Yellow)',
                2: 'Emergency Care (Red)'
            }
        }
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print("✅ Model loaded successfully!")
        
    def predict(self, image_bytes):
        """
        Predict wound assessment
        
        Args:
            image_bytes: Image file in bytes
            
        Returns:
            dict with predictions for all categories
        """
        # Load and preprocess image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0)
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(input_tensor)
        
        # Get predictions for each category
        results = {}
        for head_name, classes in self.wound_classes.items():
            num_classes = len(classes)
            classifier = nn.Linear(features.shape[1], num_classes)
            
            with torch.no_grad():
                output = classifier(features)
                probabilities = torch.softmax(output[0], dim=0)
                
                # Get top prediction
                top_idx = probabilities.argmax().item()
                top_confidence = probabilities[top_idx].item()
                
                results[head_name] = {
                    'prediction': classes[top_idx],
                    'confidence': round(top_confidence, 2)
                }
        
        # Determine overall infection status
        infection_risk = results['infection_risk']['prediction']
        is_infected = infection_risk in ['Medium', 'High']
        
        # Create summary
        summary = {
            'prediction': infection_risk + ' Infection Risk',
            'confidence': results['infection_risk']['confidence'],
            'is_infected': is_infected,
            'healing_status': results['healing_status']['prediction'],
            'urgency': results['urgency']['prediction'],
            'edema': results['edema']['prediction'],
            'erythema': results['erythema']['prediction'],
            'exudate_type': results['exudate_type']['prediction'],
            'detailed_results': results
        }
        
        print(f"🔍 Prediction: {summary['prediction']} ({summary['confidence']*100}% confidence)")
        
        return summary


# Global model instance
_model = None

def get_model():
    """Get or create the global model instance"""
    global _model
    if _model is None:
        _model = WoundModel()
    return _model