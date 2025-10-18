// Configuration
const API_URL = 'http://localhost:8000/predict'; // Change this to your backend URL
const GOOGLE_PLACES_API_KEY = 'AIzaSyDbeJQAyI2_FkgjNXtxQ2FNdOuljnCLij4'; // Get this from Google Cloud Console

let currentImage = null;
let cameraStream = null;
let userLocation = null;

// Handle file selection from upload
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file && file.type.startsWith('image/')) {
        currentImage = file;
        displayImagePreview(file);
    } else {
        alert('Please select a valid image file');
    }
}

// Display image preview
function displayImagePreview(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        const preview = document.getElementById('imagePreview');
        preview.src = e.target.result;
        
        // Show preview section, hide others
        document.getElementById('previewSection').style.display = 'block';
        document.getElementById('cameraSection').style.display = 'none';
        document.getElementById('resultsSection').style.display = 'none';
    };
    reader.readAsDataURL(file);
}

// Open camera
async function openCamera() {
    try {
        const cameraSection = document.getElementById('cameraSection');
        const video = document.getElementById('camera');
        
        cameraSection.style.display = 'block';
        document.getElementById('previewSection').style.display = 'none';
        document.getElementById('resultsSection').style.display = 'none';
        
        // Request camera access
        cameraStream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'environment' } // Try to use back camera on mobile
        });
        
        video.srcObject = cameraStream;
    } catch (error) {
        console.error('Camera access error:', error);
        alert('Could not access camera. Please make sure you have granted camera permissions.');
    }
}

// Capture photo from camera
function capturePhoto() {
    const video = document.getElementById('camera');
    const canvas = document.getElementById('canvas');
    const context = canvas.getContext('2d');
    
    // Set canvas size to match video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    // Draw video frame to canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    
    // Convert canvas to blob
    canvas.toBlob(function(blob) {
        currentImage = new File([blob], 'camera-photo.jpg', { type: 'image/jpeg' });
        displayImagePreview(currentImage);
        closeCamera();
    }, 'image/jpeg');
}

// Close camera
function closeCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
    }
    document.getElementById('cameraSection').style.display = 'none';
}

// Clear selected image
function clearImage() {
    currentImage = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('previewSection').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'none';
}

// Analyze image
async function analyzeImage() {
    if (!currentImage) {
        alert('Please select an image first');
        return;
    }
    
    // Show loading
    document.getElementById('previewSection').style.display = 'none';
    document.getElementById('loadingSection').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
    
    // Step 1: Get user location
    try {
        console.log('Requesting user location...');
        userLocation = await getUserLocation();
        console.log('Location obtained:', userLocation);
    } catch (error) {
        console.error('Location error:', error);
        
        // Use default location if geolocation fails
        userLocation = { lat: 37.3382, lon: -121.8863 }; // Default to San Jose
        console.log('Using default location:', userLocation);
        
        alert('Could not get your exact location. Using default location for hospital search. Please enable location services for better results.');
    }
    
    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', currentImage);
        formData.append('latitude', userLocation.lat);
        formData.append('longitude', userLocation.lon);
        
        // Send to backend
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        // The backend already returns hospitals in the result
        const hospitals = result.hospitals || [];
        
        // Display everything
        displayResults(result, hospitals);
        
    } catch (error) {
        console.error('Analysis error:', error);
        console.error('Error details:', {
            message: error.message,
            stack: error.stack
        });
        
        // Hide loading
        document.getElementById('loadingSection').style.display = 'none';
        
        // Show detailed error message
        let errorMsg = 'Could not connect to backend.\n\n';
        errorMsg += 'Please check:\n';
        errorMsg += '1. Backend is running at http://localhost:8000\n';
        errorMsg += '2. Run: cd backend && python3 app.py\n';
        errorMsg += '3. Check browser console for details\n\n';
        errorMsg += `Error: ${error.message}`;
        
        alert(errorMsg);
        document.getElementById('previewSection').style.display = 'block';
    }
}

// Display results
function displayResults(result, hospitals) {
    console.log('Displaying results:', result);
    console.log('Hospitals:', hospitals);
    
    document.getElementById('loadingSection').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'block';
    
    const resultsContent = document.getElementById('resultsContent');
    
    // Determine status class based on binary classification
    let statusClass = 'status-healthy';
    let statusIcon = '✅';
    
    if (result.prediction === 'Infected') {
        statusClass = 'status-infected';
        statusIcon = '⚠️';
    } else if (result.prediction === 'Not Infected') {
        statusClass = 'status-healthy';
        statusIcon = '✅';
    }
    
    // Build results HTML
    resultsContent.innerHTML = `
        <div class="result-status ${statusClass}">
            <div>${statusIcon} ${result.prediction}</div>
            <div class="confidence">Confidence: ${(result.confidence * 100).toFixed(1)}%</div>
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background: #fff3cd; border-radius: 10px; color: #856404;">
            <strong>⚕️ Medical Advice:</strong>
            ${getAdvice(result.prediction)}
        </div>
    `;
    
    // Display hospitals
    if (hospitals && hospitals.length > 0) {
        displayHospitals(hospitals);
    } else {
        console.warn('No hospitals found');
    }
}

// Get medical advice based on prediction (binary classification)
function getAdvice(prediction) {
    const advice = {
        'Not Infected': '✅ This wound does not appear to be infected. Continue regular wound care: keep it clean, dry, and covered. Monitor for signs of infection (increased redness, swelling, warmth, pus, or fever) and consult a doctor if symptoms develop.',
        'Infected': '⚠️ This wound appears to be infected. Please seek immediate medical attention from the nearest hospital or emergency room. Signs of infection include redness, swelling, warmth, pus, increased pain, or fever. Do not delay treatment.',
    };
    return advice[prediction] || 'Please consult a healthcare professional for proper evaluation.';
}

// Analyze another image
function analyzeAnother() {
    clearImage();
    document.getElementById('resultsSection').style.display = 'none';
}

// ===== LOCATION & HOSPITAL FUNCTIONS =====

// Get user's current location
function getUserLocation() {
    return new Promise((resolve, reject) => {
        if (!navigator.geolocation) {
            reject(new Error('Geolocation is not supported by your browser'));
            return;
        }
        
        navigator.geolocation.getCurrentPosition(
            (position) => {
                resolve({
                    lat: position.coords.latitude,
                    lon: position.coords.longitude
                });
            },
            (error) => {
                reject(error);
            }
        );
    });
}

// Display hospitals in the UI
function displayHospitals(hospitals) {
    const hospitalsList = document.getElementById('hospitalsList');
    hospitalsList.innerHTML = '';
    
    hospitals.slice(0, 5).forEach((hospital, index) => {
        const card = document.createElement('div');
        card.className = 'hospital-card';
        
        // Generate Google Maps directions URL
        const directionsUrl = `https://www.google.com/maps/dir/?api=1&destination=${encodeURIComponent(hospital.address)}`;
        
        card.innerHTML = `
            <div class="hospital-header">
                <div>
                    <div class="hospital-name">${index + 1}. ${hospital.name}</div>
                    <div class="hospital-address">📍 ${hospital.address}</div>
                </div>
                <div class="hospital-distance">${hospital.distance}</div>
            </div>
            <div class="hospital-info">
                <span class="hospital-phone">📞 ${hospital.phone || 'N/A'}</span>
            </div>
            <a href="${directionsUrl}" target="_blank" class="btn-directions">
                🧭 Get Directions
            </a>
        `;
        
        hospitalsList.appendChild(card);
    });
}

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    console.log('BandAId Frontend Loaded');
    console.log('Backend API URL:', API_URL);
    console.log('Google Places API Key configured:', GOOGLE_PLACES_API_KEY !== 'YOUR_API_KEY_HERE');
});
