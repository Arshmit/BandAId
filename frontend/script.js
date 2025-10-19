// Configuration
const API_URL = 'http://localhost:8000/predict'; // Change this to your backend URL

let currentImage = null;
let cameraStream = null;
let userLocation = null;

// Handle file selection from upload
function handleFileSelect(event) {
    console.log('File select triggered', event);
    try {
        const file = event.target.files[0];
        console.log('Selected file:', file);
        if (file && file.type.startsWith('image/')) {
            currentImage = file;
            displayImagePreview(file);
        } else {
            console.warn('Invalid file type selected:', file ? file.type : 'no file');
            alert('Please select a valid image file (JPEG, PNG, etc.)');
        }
    } catch (error) {
        console.error('Error in handleFileSelect:', error);
        alert('Error selecting file. Please try again.');
    }
}

// Display image preview
function displayImagePreview(file) {
    console.log('Displaying preview for file:', file.name);
    const reader = new FileReader();
    
    reader.onload = function(e) {
        console.log('File read complete');
        const preview = document.getElementById('imagePreview');
        if (!preview) {
            console.error('imagePreview element not found');
            return;
        }
        preview.src = e.target.result;
        
        // Show preview section, hide others
        const sections = {
            'previewSection': 'block',
            'cameraSection': 'none',
            'resultsSection': 'none',
            'questionnaireSection': 'none'
        };
        
        Object.entries(sections).forEach(([id, display]) => {
            const element = document.getElementById(id);
            if (element) {
                element.style.display = display;
            } else {
                console.warn(`${id} element not found`);
            }
        });
    };
    
    reader.onerror = function(error) {
        console.error('Error reading file:', error);
        alert('Error loading image preview. Please try again.');
    };
    
    try {
        reader.readAsDataURL(file);
    } catch (error) {
        console.error('Error starting file read:', error);
        alert('Error processing image. Please try a different file.');
    }
}

// Open camera
async function openCamera() {
    console.log('Opening camera...');
    try {
        const cameraSection = document.getElementById('cameraSection');
        const video = document.getElementById('camera');
        
        if (!cameraSection || !video) {
            console.error('Camera elements not found');
            alert('Could not initialize camera interface');
            return;
        }
        
        // Hide all sections except camera
        const sections = {
            'cameraSection': 'block',
            'previewSection': 'none',
            'resultsSection': 'none',
            'questionnaireSection': 'none'
        };
        
        Object.entries(sections).forEach(([id, display]) => {
            const element = document.getElementById(id);
            if (element) {
                element.style.display = display;
            }
        });
        
        console.log('Requesting camera access...');
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('Camera API not supported in this browser');
        }
        
        // Request camera access
        cameraStream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'environment' } // Try to use back camera on mobile
        });
        
        console.log('Camera access granted');
        video.srcObject = cameraStream;
        
        // Add error handler for video element
        video.onerror = function(error) {
            console.error('Video element error:', error);
            alert('Error displaying camera feed');
        };
        
    } catch (error) {
        console.error('Camera access error:', error);
        alert('Could not access camera. Please make sure you have granted camera permissions and are using a supported browser.');
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
    
    // Show questionnaire instead of sending request immediately
    document.getElementById('loadingSection').style.display = 'none';
    document.getElementById('questionnaireSection').style.display = 'block';
}

// Submit questionnaire and send everything to backend
async function submitQuestionnaire() {
    // Show loading
    document.getElementById('questionnaireSection').style.display = 'none';
    document.getElementById('loadingSection').style.display = 'block';
    
    try {
        // Get questionnaire answers
        const q1Answer = document.querySelector('input[name="q1"]:checked')?.value || 'no';
        const q2Answer = document.getElementById('q2').value;
        const q3Answer = document.querySelector('input[name="q3"]:checked')?.value || 'no';

        // Create form data with image, location, and questionnaire
        const formData = new FormData();
        formData.append('file', currentImage);
        formData.append('latitude', userLocation.lat);
        formData.append('longitude', userLocation.lon);
        formData.append('q1_hot_touch', q1Answer);
        formData.append('q2_pain_level', q2Answer);
        formData.append('q3_wheezing', q3Answer);
        
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
        document.getElementById('questionnaireSection').style.display = 'block';
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

// Get medical advice based on prediction and risk factors
function getAdvice(prediction) {
    // Get questionnaire answers
    const painLevel = parseInt(document.getElementById('q2').value) || 0;
    const isHot = document.querySelector('input[name="q1"]:checked')?.value === 'yes';
    const isWheezing = document.querySelector('input[name="q3"]:checked')?.value === 'yes';
    
    // Only show medical advice if infected or if there are risk factors
    if (prediction === 'Infected' || painLevel >= 5 || isHot || isWheezing) {
        return '⚠️ Please seek immediate medical attention from the nearest hospital or emergency room. Signs of infection include redness, swelling, warmth, pus, increased pain, or fever. Do not delay treatment.';
    } else {
        return '✅ Based on the image analysis, no immediate medical attention appears necessary.';
    }
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
    // `hospitals` is expected to be an array of objects with {name, address}
    hospitals.slice(0, 3).forEach((hospital, index) => {
        const card = document.createElement('div');
        card.className = 'hospital-card';

        const directionsUrl = `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(hospital.address)}`;

        card.innerHTML = `
            <div class="hospital-header">
                <div>
                    <div class="hospital-name">${index + 1}. ${hospital.name}</div>
                    <div class="hospital-address">📍 ${hospital.address}</div>
                </div>
            </div>
            <a href="${directionsUrl}" target="_blank" class="btn-directions">
                🧭 Open in Maps
            </a>
        `;

        hospitalsList.appendChild(card);
    });
}

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    console.log('BandAId Frontend Loaded');
    console.log('Backend API URL:', API_URL);
    
    // Setup pain scale value display
    const painScale = document.getElementById('q2');
    const painValue = document.getElementById('q2Value');
    if (painScale && painValue) {
        painScale.addEventListener('input', function() {
            painValue.textContent = this.value;
        });
    }
});