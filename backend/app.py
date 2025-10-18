"""
BandAId Backend API
Flask server for wound infection detection + hospital finder
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from model import get_model
from hospital_finder import find_nearby_hospitals
import os

app = Flask(__name__)

# Enable CORS for frontend to connect
CORS(app)

# Load AI model at startup
print("🚀 Starting BandAId Backend...")
model = get_model()
print("✅ Backend ready!")


@app.route('/', methods=['GET'])
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "message": "BandAId Backend API",
        "endpoints": {
            "/predict": "POST - Upload wound image for analysis",
            "/health": "GET - Check server health"
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Check if server is healthy"""
    return jsonify({"status": "healthy"})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint
    Accepts: multipart/form-data with 'file', 'latitude', 'longitude'
    Returns: JSON with prediction and nearby hospitals
    """
    try:
        # Validate file is present
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400
        
        # Get location (optional)
        try:
            latitude = float(request.form.get('latitude', 37.7749))  # Default to SF
            longitude = float(request.form.get('longitude', -122.4194))
        except ValueError:
            return jsonify({"error": "Invalid latitude/longitude"}), 400
        
        print(f"\n📍 Request from location: ({latitude}, {longitude})")
        
        # Read image
        image_bytes = file.read()
        
        # ===== STEP 1: AI Model Prediction =====
        print("🔬 Running AI model...")
        prediction_result = model.predict(image_bytes)
        
        # ===== STEP 2: Find Nearby Hospitals =====
        print("🏥 Finding nearby hospitals...")
        is_infected = prediction_result.get('is_infected', False)
        hospitals = find_nearby_hospitals(
            latitude, 
            longitude, 
            is_infected=is_infected,
            max_results=5
        )
        
        # ===== STEP 3: Build Response =====
        response = {
            "prediction": prediction_result['prediction'],
            "confidence": prediction_result['confidence'],
            "is_infected": is_infected,
            "hospitals": hospitals,
            "user_location": {
                "lat": latitude,
                "lon": longitude
            },
            "advice": get_medical_advice(is_infected)
        }
        
        print(f"✅ Response ready: {prediction_result['prediction']}, {len(hospitals)} hospitals found")
        
        return jsonify(response), 200
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500


def get_medical_advice(is_infected):
    """
    Get medical advice based on infection status
    """
    if is_infected:
        return (
            "⚠️ This wound appears to be infected. "
            "Please seek immediate medical attention from the nearest hospital or emergency room. "
            "Signs of infection include redness, swelling, warmth, pus, or fever."
        )
    else:
        return (
            "✅ This wound does not appear to be infected. "
            "Continue regular wound care: keep it clean, dry, and covered. "
            "Monitor for signs of infection and consult a doctor if symptoms develop."
        )


if __name__ == '__main__':
    print("\n" + "="*50)
    print("🩹 BandAId Backend Server")
    print("="*50)
    print("\n📡 Starting server on http://localhost:8000")
    print("🔗 Frontend should connect to: http://localhost:8000/predict")
    print("\n💡 Press Ctrl+C to stop\n")
    
    app.run(
        host='0.0.0.0',
        port=8000,
        debug=True
    )
