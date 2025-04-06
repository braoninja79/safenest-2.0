from flask import Flask, Response, render_template_string
import cv2
import numpy as np
import os
from pathlib import Path
from models import load_models, MODEL_MEAN_VALUES, GENDER_LIST

app = Flask(__name__)

# Environment variables for configuration
PORT = int(os.environ.get('PORT', 8000))
DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'

# Initialize video capture
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open video source")
except Exception as e:
    print(f"Error initializing video capture: {e}")

# Load models
try:
    faceNet, genderNet = load_models()
except Exception as e:
    print(f"Error loading models: {e}")

padding = 20

def getFaceBox(net, frame, conf_threshold=0.7):
    try:
        frameOpencvDnn = frame.copy()
        frameHeight = frameOpencvDnn.shape[0]
        frameWidth = frameOpencvDnn.shape[1]
        blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
        net.setInput(blob)
        detections = net.forward()
        bboxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frameWidth)
                y1 = int(detections[0, 0, i, 4] * frameHeight)
                x2 = int(detections[0, 0, i, 5] * frameWidth)
                y2 = int(detections[0, 0, i, 6] * frameHeight)
                bboxes.append([x1, y1, x2, y2])
                cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight/150)), 8)
        return frameOpencvDnn, bboxes
    except Exception as e:
        print(f"Error in face detection: {e}")
        return frame, []

def check_coverage(frame, grid_rows=3, grid_cols=3, variance_threshold=100.0):
    try:
        height, width = frame.shape[:2]
        cell_height = height // grid_rows
        cell_width = width // grid_cols
        covered_cells = 0
        total_cells = grid_rows * grid_cols
        
        for i in range(grid_rows):
            for j in range(grid_cols):
                y1 = i * cell_height
                y2 = (i + 1) * cell_height if i < grid_rows - 1 else height
                x1 = j * cell_width
                x2 = (j + 1) * cell_width if j < grid_cols - 1 else width
                cell = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                var = laplacian.var()
                if var < variance_threshold:
                    covered_cells += 1
        
        return covered_cells / total_cells
    except Exception as e:
        print(f"Error in coverage detection: {e}")
        return 0.0

def generate_frames():
    while True:
        try:
            success, frame = cap.read()
            if not success:
                print("Failed to grab frame")
                break

            # Process frame with face detection
            frameFace, bboxes = getFaceBox(faceNet, frame)
            
            numMales = 0
            numFemales = 0
            numPersons = len(bboxes)
            
            # Process each detected face
            for bbox in bboxes:
                try:
                    face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),
                               max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
                    
                    # Gender detection
                    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                    genderNet.setInput(blob)
                    genderPreds = genderNet.forward()
                    gender = GENDER_LIST[genderPreds[0].argmax()]
                    
                    # Add gender label to frame
                    label = f"{gender}"
                    cv2.putText(frameFace, label, (bbox[0], bbox[1]-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
                    
                    # Update counts
                    if gender == 'Male':
                        numMales += 1
                    else:
                        numFemales += 1
                except Exception as e:
                    print(f"Error processing face: {e}")
                    continue
            
            # Determine status
            status = "No person detected"
            if numPersons >= 1:
                if numFemales == 1:
                    if numMales >= 3:
                        status = "Warning: Woman is surrounded by men"
                    else:
                        status = "Woman is alone"
                elif numPersons >= 2:
                    status = "Multiple persons detected"
            
            # Check coverage
            coverage_ratio = check_coverage(frame)
            coverage_status = f"Coverage: {coverage_ratio:.2f}"
            if coverage_ratio >= 0.99:
                coverage_status = "Warning: 100% display is covered!"
            elif coverage_ratio >= 0.4:
                coverage_status = "Warning: Screen covered over 40%!"
            
            # Add overlay text to frame
            cv2.putText(frameFace, f"Persons: {numPersons}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frameFace, f"Males: {numMales}", (10, 60),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frameFace, f"Females: {numFemales}", (10, 90),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(frameFace, status, (10, 120),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frameFace, coverage_status, (10, 150),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Convert frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frameFace)
            if not ret:
                print("Failed to encode frame")
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"Error in frame generation: {e}")
            continue

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SafeNest - Women Safety App</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
    <style>
        :root {
            --primary-color: #8e44ad;
            --secondary-color: #9b59b6;
            --accent-color: #d5b8ff;
            --danger-color: #e74c3c;
            --light-bg: #f8f9fa;
        }
        
        body {
            background: linear-gradient(135deg, var(--light-bg) 0%, #ffffff 100%);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            min-height: 100vh;
        }
        
        .navbar {
            background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            padding: 1rem 0;
            box-shadow: 0 2px 15px rgba(0, 0, 0, 0.1);
        }
        
        .navbar-brand {
            font-size: 1.5rem;
            font-weight: bold;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .navbar-brand i {
            color: var(--accent-color);
            margin-right: 8px;
        }

        .hero-section {
            background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), url('https://source.unsplash.com/1600x900/?safety,protection');
            background-size: cover;
            background-position: center;
            color: white;
            padding: 4rem 0;
            margin-bottom: 2rem;
            text-align: center;
        }

        .hero-section h1 {
            font-size: 3rem;
            font-weight: bold;
            margin-bottom: 1rem;
        }

        .hero-section p {
            font-size: 1.2rem;
            max-width: 800px;
            margin: 0 auto;
        }
        
        .video-container {
            max-width: 900px;
            margin: 2rem auto;
            border-radius: 15px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
            border: 3px solid #fff;
            position: relative;
            background: #000;
        }
        
        #videoStream {
            width: 100%;
            height: auto;
            display: block;
            transition: transform 0.3s ease;
        }
        
        .controls-container {
            text-align: center;
            margin: 2rem 0;
        }
        
        .btn-emergency {
            background: var(--danger-color);
            border: none;
            padding: 1rem 2rem;
            font-size: 1.2rem;
            border-radius: 50px;
            box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
            transition: all 0.3s ease;
        }
        
        .btn-emergency:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(231, 76, 60, 0.4);
            background: #c0392b;
        }
        
        .btn-emergency i {
            margin-right: 8px;
        }
        
        .card {
            border: none;
            border-radius: 15px;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
            overflow: hidden;
            transition: transform 0.3s ease;
            height: 100%;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        .card-header {
            background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            padding: 1.2rem;
            border-bottom: none;
        }
        
        .card-header h5 {
            font-weight: 600;
            letter-spacing: 0.5px;
        }
        
        .emergency-contact {
            background: var(--light-bg);
            border-radius: 12px;
            padding: 1.5rem;
        }
        
        .contact-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.8rem;
            margin-bottom: 0.5rem;
            border-radius: 8px;
            background: white;
            transition: all 0.3s ease;
        }
        
        .contact-item:hover {
            transform: translateX(5px);
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }
        
        .contact-info {
            font-size: 1.1rem;
        }
        
        .contact-info strong {
            color: var(--primary-color);
        }
        
        .btn-call {
            background: transparent;
            color: var(--primary-color);
            border: 2px solid var(--primary-color);
            border-radius: 50px;
            padding: 0.5rem 1.2rem;
            transition: all 0.3s ease;
        }
        
        .btn-call:hover {
            background: var(--primary-color);
            color: white;
            transform: scale(1.05);
        }
        
        .alert {
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }

        .feature-card {
            text-align: center;
            padding: 2rem;
        }

        .feature-icon {
            font-size: 2.5rem;
            color: var(--primary-color);
            margin-bottom: 1rem;
        }

        .safety-tips {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            margin-top: 3rem;
            box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
        }

        .tip-item {
            display: flex;
            align-items: flex-start;
            margin-bottom: 1.5rem;
            padding: 1rem;
            background: var(--light-bg);
            border-radius: 10px;
            transition: all 0.3s ease;
        }

        .tip-item:hover {
            transform: translateX(5px);
            background: white;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }

        .tip-icon {
            font-size: 1.5rem;
            color: var(--primary-color);
            margin-right: 1rem;
        }

        .footer {
            background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: white;
            padding: 2rem 0;
            margin-top: 3rem;
        }

        .footer-content {
            text-align: center;
        }

        .social-links {
            margin-top: 1rem;
        }

        .social-links a {
            color: white;
            font-size: 1.5rem;
            margin: 0 10px;
            transition: all 0.3s ease;
        }

        .social-links a:hover {
            color: var(--accent-color);
            transform: translateY(-3px);
        }
        
        @media (max-width: 768px) {
            .video-container {
                margin: 1rem;
                border-radius: 10px;
            }
            
            .card {
                margin: 1rem;
            }

            .hero-section {
                padding: 2rem 0;
            }

            .hero-section h1 {
                font-size: 2rem;
            }
        }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark">
        <div class="container">
            <a class="navbar-brand" href="#">
                <i class="bi bi-shield-check"></i>
                SafeNest
            </a>
        </div>
    </nav>

    <div class="hero-section">
        <div class="container">
            <h1>Welcome to SafeNest</h1>
            <p>Your personal safety companion powered by advanced AI technology. We're here to keep you safe and secure.</p>
        </div>
    </div>

    <main class="container py-4">
        <div class="row mb-5">
            <div class="col-md-4 mb-4">
                <div class="card feature-card">
                    <div class="feature-icon">
                        <i class="bi bi-camera-video"></i>
                    </div>
                    <h3>Real-time Monitoring</h3>
                    <p>Advanced AI-powered video monitoring system for instant threat detection.</p>
                </div>
            </div>
            <div class="col-md-4 mb-4">
                <div class="card feature-card">
                    <div class="feature-icon">
                        <i class="bi bi-gender-female"></i>
                    </div>
                    <h3>Gender Detection</h3>
                    <p>Intelligent gender recognition to identify potential risk situations.</p>
                </div>
            </div>
            <div class="col-md-4 mb-4">
                <div class="card feature-card">
                    <div class="feature-icon">
                        <i class="bi bi-telephone-fill"></i>
                    </div>
                    <h3>Quick Response</h3>
                    <p>Instant access to emergency services when you need them most.</p>
                </div>
            </div>
        </div>

        <div class="video-container">
            <img id="videoStream" src="{{ url_for('video_feed') }}" alt="Video Feed">
        </div>

        <div class="controls-container">
            <button class="btn btn-emergency" id="emergencyBtn">
                <i class="bi bi-telephone-fill"></i>
                Emergency Call
            </button>
        </div>

        <div class="row justify-content-center mt-4">
            <div class="col-md-8 col-lg-6">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0 text-white">
                            <i class="bi bi-telephone-outbound me-2"></i>
                            Emergency Contacts
                        </h5>
                    </div>
                    <div class="card-body">
                        <div class="emergency-contact">
                            <div class="contact-item">
                                <div class="contact-info">
                                    <strong>Police:</strong> 100
                                </div>
                                <button class="btn btn-call" onclick="window.location.href='tel:100'">
                                    <i class="bi bi-telephone"></i> Call
                                </button>
                            </div>
                            <div class="contact-item">
                                <div class="contact-info">
                                    <strong>Women Helpline:</strong> 1091
                                </div>
                                <button class="btn btn-call" onclick="window.location.href='tel:1091'">
                                    <i class="bi bi-telephone"></i> Call
                                </button>
                            </div>
                            <div class="contact-item">
                                <div class="contact-info">
                                    <strong>Emergency Services:</strong> 112
                                </div>
                                <button class="btn btn-call" onclick="window.location.href='tel:112'">
                                    <i class="bi bi-telephone"></i> Call
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="safety-tips">
            <h2 class="text-center mb-4">Safety Tips</h2>
            <div class="row">
                <div class="col-md-6">
                    <div class="tip-item">
                        <i class="bi bi-geo-alt tip-icon"></i>
                        <div>
                            <h5>Share Your Location</h5>
                            <p>Always share your location with trusted contacts when traveling alone.</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="tip-item">
                        <i class="bi bi-phone tip-icon"></i>
                        <div>
                            <h5>Keep Phone Charged</h5>
                            <p>Ensure your phone is always charged and carry a power bank.</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="tip-item">
                        <i class="bi bi-people tip-icon"></i>
                        <div>
                            <h5>Stay in Groups</h5>
                            <p>Try to stay in well-lit areas and travel in groups when possible.</p>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="tip-item">
                        <i class="bi bi-shield-check tip-icon"></i>
                        <div>
                            <h5>Trust Your Instincts</h5>
                            <p>If something feels wrong, trust your gut and seek help immediately.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <footer class="footer">
        <div class="container">
            <div class="footer-content">
                <h3>SafeNest</h3>
                <p>Empowering women with safety technology</p>
                <div class="social-links">
                    <a href="#"><i class="bi bi-facebook"></i></a>
                    <a href="#"><i class="bi bi-twitter"></i></a>
                    <a href="#"><i class="bi bi-instagram"></i></a>
                    <a href="#"><i class="bi bi-linkedin"></i></a>
                </div>
            </div>
        </div>
    </footer>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const videoStream = document.getElementById('videoStream');
            const emergencyBtn = document.getElementById('emergencyBtn');

            emergencyBtn.addEventListener('click', function() {
                const audio = new Audio('https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg');
                audio.play().catch(err => console.error('Error playing alarm:', err));
                
                if (confirm('Do you want to call emergency services?')) {
                    window.location.href = 'tel:112';
                }
            });

            videoStream.addEventListener('error', function() {
                const alertHtml = `
                    <div class="alert alert-danger alert-dismissible fade show mt-3" role="alert">
                        <strong>Connection Error!</strong> Unable to connect to video stream. Please check if the server is running.
                        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
                    </div>
                `;
                document.querySelector('.video-container').insertAdjacentHTML('afterend', alertHtml);
            });
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.errorhandler(404)
def not_found_error(error):
    return render_template_string("""
        <div style="text-align: center; padding: 50px;">
            <h1>404 - Page Not Found</h1>
            <p>The requested page could not be found.</p>
            <a href="/" class="btn btn-primary">Return to Home</a>
        </div>
    """), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template_string("""
        <div style="text-align: center; padding: 50px;">
            <h1>500 - Internal Server Error</h1>
            <p>Something went wrong on our end. Please try again later.</p>
            <a href="/" class="btn btn-primary">Return to Home</a>
        </div>
    """), 500

def cleanup():
    print("Cleaning up resources...")
    if cap is not None:
        cap.release()

import atexit
atexit.register(cleanup)

if __name__ == '__main__':
    try:
        # Vercel requires port 3000
        port = int(os.environ.get('PORT', 3000))
        app.run(host='0.0.0.0', port=port, debug=DEBUG)
    except Exception as e:
        print(f"Error starting server: {e}")
        cleanup()
