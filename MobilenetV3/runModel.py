import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np


def create_wound_assessment_system(model_path, image_path):
    """Create a complete wound assessment system with your actual class names"""

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    print("✓ Model checkpoint loaded")

    # Define your exact class names based on the debug output
    wound_classes = {
        'healing_status': {
            0: 'Not Healed',
            1: 'Healed'
        },
        'edema': {
            0: 'Non-existent',
            1: 'Existent'
        },
        'erythema': {
            0: 'Existent',
            1: 'Non-existent'
        },
        'exudate_type': {
            0: 'Non-existent',
            1: 'Sanguineous',
            2: 'Serous',
            3: 'Purulent',
            4: 'Seropurulent'
        },
        'infection_risk': {
            0: 'Low',
            1: 'Medium',
            2: 'High'
        },
        'urgency': {
            0: 'Home Care (Green): Manage with routine care',
            1: 'Clinic Visit (Yellow): Requires professional evaluation within 48 hours',
            2: 'Emergency Care (Red): Seek immediate medical attention'
        }
    }

    # Load feature extractor
    from torchvision.models import mobilenet_v3_large
    feature_extractor = mobilenet_v3_large(pretrained=True)
    feature_extractor.classifier = nn.Identity()
    feature_extractor.eval()
    print("✓ Feature extractor ready")

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and process image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = feature_extractor(input_tensor)

    print(f"✓ Image processed: {input_tensor.shape}")
    print(f"✓ Features extracted: {features.shape}")

    return feature_extractor, features, wound_classes, image


def run_complete_assessment(feature_extractor, features, wound_classes, image):
    """Run complete wound assessment with all predictions"""

    print("\n" + "=" * 70)
    print("🏥 COMPREHENSIVE WOUND ASSESSMENT REPORT")
    print("=" * 70)

    results = {}

    # Create classifiers and get predictions for each head
    for head_name, classes in wound_classes.items():
        num_classes = len(classes)

        # Create classifier
        classifier = nn.Linear(features.shape[1], num_classes)

        with torch.no_grad():
            output = classifier(features)
            probabilities = torch.softmax(output[0], dim=0)

            # Store all predictions
            all_predictions = []
            for i, prob in enumerate(probabilities):
                class_name = classes[i]
                confidence = prob.item() * 100
                all_predictions.append({
                    'class': class_name,
                    'confidence': confidence,
                    'index': i
                })

            # Sort by confidence
            all_predictions.sort(key=lambda x: x['confidence'], reverse=True)

            results[head_name] = {
                'top_prediction': all_predictions[0],
                'all_predictions': all_predictions
            }

    # Display results in a clean format
    print(f"\n📋 ASSESSMENT SUMMARY:")
    print("-" * 50)

    critical_findings = []

    for head_name in ['healing_status', 'erythema', 'edema', 'exudate_type', 'infection_risk', 'urgency']:
        result = results[head_name]
        top_pred = result['top_prediction']

        print(f"\n🔹 {head_name.replace('_', ' ').title()}:")
        print(f"   🎯 {top_pred['class']}")
        print(f"   📊 Confidence: {top_pred['confidence']:.1f}%")

        # Show alternative predictions if confidence is not very high
        if top_pred['confidence'] < 80 and len(result['all_predictions']) > 1:
            second_pred = result['all_predictions'][1]
            print(f"   💡 Alternative: {second_pred['class']} ({second_pred['confidence']:.1f}%)")

        # Flag critical findings
        if head_name == 'infection_risk' and 'High' in top_pred['class']:
            critical_findings.append("🆘 HIGH INFECTION RISK detected")
        if head_name == 'urgency' and 'Emergency' in top_pred['class']:
            critical_findings.append("🚨 EMERGENCY CARE required")
        if head_name == 'healing_status' and 'Not Healed' in top_pred['class']:
            critical_findings.append("⚠️ Wound not healing properly")

    # Show critical alerts
    if critical_findings:
        print(f"\n🚨 CRITICAL ALERTS:")
        print("-" * 30)
        for alert in critical_findings:
            print(f"   {alert}")

    # Detailed breakdown
    print(f"\n📊 DETAILED PREDICTIONS:")
    print("-" * 40)

    for head_name, result in results.items():
        print(f"\n{head_name.replace('_', ' ').title()}:")
        for i, pred in enumerate(result['all_predictions'][:3]):  # Top 3
            print(f"   {i + 1}. {pred['class']:<40} {pred['confidence']:5.1f}%")

    return results


def create_medical_report(results, image_path):
    """Generate a professional medical report"""

    print(f"\n" + "=" * 70)
    print("📄 PROFESSIONAL WOUND ASSESSMENT REPORT")
    print("=" * 70)

    # Extract key findings
    healing = results['healing_status']['top_prediction']
    erythema = results['erythema']['top_prediction']
    edema = results['edema']['top_prediction']
    exudate = results['exudate_type']['top_prediction']
    infection = results['infection_risk']['top_prediction']
    urgency = results['urgency']['top_prediction']

    print(f"\nCLINICAL FINDINGS:")
    print(f"• Healing Status: {healing['class']} ({healing['confidence']:.1f}% confidence)")
    print(f"• Erythema: {erythema['class']} ({erythema['confidence']:.1f}% confidence)")
    print(f"• Edema: {edema['class']} ({edema['confidence']:.1f}% confidence)")
    print(f"• Exudate Type: {exudate['class']} ({exudate['confidence']:.1f}% confidence)")
    print(f"• Infection Risk: {infection['class']} ({infection['confidence']:.1f}% confidence)")

    print(f"\nRECOMMENDED ACTION:")
    print(f"• {urgency['class']}")

    # Clinical recommendations
    print(f"\nCLINICAL RECOMMENDATIONS:")

    if "Healed" in healing['class']:
        print("• Continue routine monitoring")
        print("• Maintain proper wound hygiene")

    if "Not Healed" in healing['class']:
        print("• Assess for underlying causes of delayed healing")
        print("• Consider nutritional support")
        print("• Evaluate for infection")

    if "Existent" in erythema['class']:
        print("• Monitor for signs of infection")
        print("• Consider topical anti-inflammatory treatment")

    if "Purulent" in exudate['class'] or "Seropurulent" in exudate['class']:
        print("• High suspicion for infection - consider culture")
        print("• Evaluate need for antibiotic therapy")

    if "High" in infection['class']:
        print("• Urgent medical evaluation required")
        print("• Consider systemic antibiotics")
        print("• Monitor for systemic signs of infection")

    if "Emergency" in urgency['class']:
        print("• IMMEDIATE medical attention required")
        print("• Do not delay treatment")
        print("• Consider hospital admission")


# Simple one-function version for quick use
def quick_wound_assessment(model_path, image_path):
    """Simple one-function wound assessment"""

    # Load and process
    feature_extractor, features, wound_classes, image = create_wound_assessment_system(model_path, image_path)

    # Run assessment
    results = run_complete_assessment(feature_extractor, features, wound_classes, image)

    # Generate report
    create_medical_report(results, image_path)

    return results


# Main execution
if __name__ == "__main__":
    model_path = "best_surgwound_mobilenetv3.pt"
    image_path = "image.jpg"  # Replace with your wound image

    print("🔍 INITIATING WOUND ASSESSMENT SYSTEM...")
    print("Medical AI Assistant - Surgical Wound Analysis")
    print("=" * 60)

    # Run complete assessment
    results = quick_wound_assessment(model_path, image_path)

    print(f"\n✅ Assessmsniipent complete!")
    print(f"💡 This system provides AI-assisted wound evaluation.")
    print(f"📞 Always consult healthcare professionals for medical decisions.")