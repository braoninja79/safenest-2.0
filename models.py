import cv2
import numpy as np
import urllib.request
import os
from pathlib import Path

MODEL_URLS = {
    'face_proto': 'YOUR_RAW_GITHUB_URL/opencv_face_detector.pbtxt',
    'face_model': 'YOUR_RAW_GITHUB_URL/opencv_face_detector_uint8.pb',
    'gender_proto': 'YOUR_RAW_GITHUB_URL/gender_deploy.prototxt',
    'gender_model': 'YOUR_RAW_GITHUB_URL/gender_net.caffemodel'
}

def download_model(url, filename):
    """Download model file if not present"""
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filename)
            print(f"Downloaded {filename}")
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            return False
    return True

def load_models():
    """Load all required models"""
    models_dir = Path(__file__).parent / 'models'
    models_dir.mkdir(exist_ok=True)
    
    model_paths = {
        'face_proto': models_dir / 'opencv_face_detector.pbtxt',
        'face_model': models_dir / 'opencv_face_detector_uint8.pb',
        'gender_proto': models_dir / 'gender_deploy.prototxt',
        'gender_model': models_dir / 'gender_net.caffemodel'
    }
    
    # Download models if needed
    for key, path in model_paths.items():
        if not download_model(MODEL_URLS[key], str(path)):
            raise RuntimeError(f"Failed to download {key}")
    
    try:
        # Load models
        faceNet = cv2.dnn.readNet(
            str(model_paths['face_model']),
            str(model_paths['face_proto'])
        )
        genderNet = cv2.dnn.readNet(
            str(model_paths['gender_model']),
            str(model_paths['gender_proto'])
        )
        
        return faceNet, genderNet
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

# Constants
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LIST = ['Male', 'Female']
