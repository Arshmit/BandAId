"""
AI Model for Wound Assessment using MobileNetV3
Loads trained multi-head classification model
"""
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large
from PIL import Image
import io
import os


class MultiHeadMobileNetV3(nn.Module):
    """Multi-head MobileNetV3 model for wound assessment (matches training architecture)"""
    
    def __init__(self, label2idx, use_large=False, pretrained=True):
        super().__init__()
        
        # Load MobileNetV3 backbone
        if use_large:
            self.backbone = mobilenet_v3_large(pretrained=pretrained)
        else:
            self.backbone = mobilenet_v3_small(pretrained=pretrained)
        
        # Use features from backbone
        self.features = self.backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Get feature dimension
        if hasattr(self.backbone, 'classifier'):
            try:
                in_features = self.backbone.classifier[0].in_features
            except:
                in_features = 576 if not use_large else 960
        else:
            in_features = 576 if not use_large else 960
        
        # Shared FC layer
        self.shared_fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Create classification heads for each task
        self.heads = nn.ModuleDict()
        for key in label2idx:
            n_classes = len(label2idx[key])
            self.heads[key] = nn.Linear(512, n_classes)
    
    def forward(self, x):
        # Extract features
        x = self.features(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.shared_fc(x)
        
        # Get predictions from each head
        outputs = {}
        for key, head in self.heads.items():
            outputs[key] = head(x)
        
        return outputs


class WoundModel:
    """Wound assessment model using trained MobileNetV3"""
    
    def __init__(self):
        """Initialize and load the trained model"""
        print("🤖 Loading trained MobileNetV3 wound assessment model...")
        
        # Path to trained model checkpoint
        model_path = os.path.join(os.path.dirname(__file__), '..', 'MobilenetV3', 'best_surgwound_mobilenetv3.pt')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
        
        # Load checkpoint (weights_only=False is safe for your own trained model)
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print("✓ Checkpoint loaded")
        
        # Get label mappings from checkpoint
        label_mappings = checkpoint.get('label_mappings', {})
        self.label2idx = label_mappings.get('label2idx', {})
        self.idx2label = label_mappings.get('idx2label', {})
        
        # Create model architecture
        self.model = MultiHeadMobileNetV3(self.label2idx, use_large=False, pretrained=False)
        
        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        print("✓ Model weights loaded")
        
        # Image preprocessing (same as training)
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✅ Model loaded successfully! Tasks: {list(self.label2idx.keys())}")
        
    def predict(self, image_bytes):
        """
        Predict wound assessment using trained model
        
        Args:
            image_bytes: Image file in bytes
            
        Returns:
            dict with predictions for all categories
        """
        # Load and preprocess image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0)
        
        # Get predictions from model
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        # Process predictions for each head
        results = {}
        for head_name, logits in outputs.items():
            probabilities = torch.softmax(logits[0], dim=0)
            
            # Get top prediction
            top_idx = probabilities.argmax().item()
            top_confidence = probabilities[top_idx].item()
            
            # Get class name from idx2label
            class_name = self.idx2label[head_name][top_idx]
            
            results[head_name] = {
                'prediction': class_name,
                'confidence': round(top_confidence, 2)
            }
        
        # Determine overall infection status
        infection_risk = results.get('infection_risk', {}).get('prediction', 'Unknown')
        is_infected = 'High' in infection_risk or 'Medium' in infection_risk
        
        # Create summary
        summary = {
            'prediction': infection_risk + ' Infection Risk',
            'confidence': results.get('infection_risk', {}).get('confidence', 0.0),
            'is_infected': is_infected,
            'healing_status': results.get('healing_status', {}).get('prediction', 'Unknown'),
            'urgency': results.get('urgency', {}).get('prediction', 'Unknown'),
            'edema': results.get('edema', {}).get('prediction', 'Unknown'),
            'erythema': results.get('erythema', {}).get('prediction', 'Unknown'),
            'exudate_type': results.get('exudate_type', {}).get('prediction', 'Unknown'),
            'detailed_results': results
        }
        
        print(f"🔍 Prediction: {summary['prediction']} ({summary['confidence']*100:.1f}% confidence)")
        
        return summary


# Global model instance
_model = None

def get_model():
    """Get or create the global model instance"""
    global _model
    if _model is None:
        _model = WoundModel()
    return _model