const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const clickBtn = document.getElementById("clickBtn");
const submitBtn = document.getElementById("submitBtn");
const errorMsg = document.getElementById("errorMsg");

let selectedFile = null;

uploadBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const validTypes = ["image/jpeg", "image/png", "image/jpg"];
  const maxSize = 5 * 1024 * 1024; // 5MB limit

  if (!validTypes.includes(file.type)) {
    showError(" Please upload a valid image file (JPG or PNG).");
    selectedFile = null;
  } else if (file.size > maxSize) {
    showError("️ File too large! Must be under 5MB.");
    selectedFile = null;
  } else {
    errorMsg.textContent = "";
    selectedFile = file;
    alert(` ${file.name} selected successfully.`);
  }
});

clickBtn.addEventListener("click", () => {
  alert(" Camera feature coming soon!");
});

submitBtn.addEventListener("click", () => {
  if (!selectedFile) {
    showError("Please upload a valid image before submitting.");
    return;
  }
  alert(`Submitting file: ${selectedFile.name}`);
});

function showError(message) {
  errorMsg.textContent = message;
  setTimeout(() => {
    errorMsg.textContent = "";
  }, 4000);
}

// Configuration
const API_URL = 'http://localhost:8000/predict'; // Change this to your backend URL
<<<<<<< Updated upstream
let currentImage = null;
=======
const GOOGLE_PLACES_API_KEY = 'AIzaSyDbeJQAyI2_FkgjNXtxQ2FNdOuljnCLij4'; // Get this from Google Cloud Console

let currentImage;
>>>>>>> Stashed changes
let cameraStream = null;

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
    
    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', currentImage);
        
        // Send to backend
        const response = await fetch(API_URL, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        displayResults(result);
        
    } catch (error) {
        console.error('Analysis error:', error);
        
        // Hide loading
        document.getElementById('loadingSection').style.display = 'none';
        
        // Show demo results 
        alert('Could not connect to backend. Showing demo results instead.');
        displayDemoResults();
    }
}

// Display results
function displayResults(result) {
    document.getElementById('loadingSection').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'block';
    
    const resultsContent = document.getElementById('resultsContent');
    
    // Determine status class
    let statusClass = 'status-healthy';
    let statusIcon = '✅';
    if (result.prediction === 'At Risk') {
        statusClass = 'status-at-risk';
        statusIcon = '⚡';
    } else if (result.prediction === 'Infected') {
        statusClass = 'status-infected';
        statusIcon = '⚠️';
    }
    
    // Build results HTML
    resultsContent.innerHTML = `
        <div class="result-status ${statusClass}">
            <div>${statusIcon} ${result.prediction}</div>
            <div class="confidence">Confidence: ${(result.confidence * 100).toFixed(1)}%</div>
        </div>
        
        <div class="probabilities">
            <h4>Detailed Analysis:</h4>
            ${Object.entries(result.probabilities).map(([label, prob]) => `
                <div class="probability-item">
                    <span>${label}</span>
                    <span><strong>${(prob * 100).toFixed(1)}%</strong></span>
                </div>
            `).join('')}
        </div>
        
        <div style="margin-top: 20px; padding: 15px; background: #fff3cd; border-radius: 10px; color: #856404;">
            <strong>⚕️ Medical Advice:</strong>
            ${getAdvice(result.prediction)}
        </div>
    `;
}

// Display demo results (for testing without backend)
function displayDemoResults() {
    const demoResult = {
        prediction: 'At Risk',
        confidence: 0.00,
        probabilities: {
            'Healthy': 0.0,
            'At Risk': 0.0,
            'Infected': 0.00
        }
    };
    displayResults(demoResult);
}

// Get medical advice based on prediction
function getAdvice(prediction) {
    const advice = {
        'Healthy': 'X',
        'At Risk': 'X',
        'Infected': 'X',
    };
    return advice[prediction] || 'Please consult a healthcare professional for proper evaluation.';
}

// Analyze another image
function analyzeAnother() {
    clearImage();
    document.getElementById('resultsSection').style.display = 'none';
}

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    console.log('BandAId Frontend Loaded');
    console.log('Backend API URL:', API_URL);
});
