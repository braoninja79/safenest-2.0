# SafeNest - Women Safety Application 🛡️

SafeNest is an AI-powered women's safety application that provides real-time video monitoring, gender detection, and safety alerts. Built with modern web technologies and machine learning capabilities, it aims to enhance women's safety through smart surveillance and quick response mechanisms.

## 🌟 Features

### 🎥 Real-time Monitoring
- Live video feed processing
- Instant face detection
- Gender recognition system
- Coverage detection for camera obstruction

### 🚨 Safety Alerts
- Detects when a woman is surrounded by multiple men
- Monitors suspicious activities
- Alerts for camera coverage/obstruction
- Real-time status updates

### 💡 Smart Features
- Gender-based detection and counting
- Multiple person detection
- Camera obstruction detection
- Intuitive user interface

### 🎯 Additional Features
- Emergency contact system
- Quick response buttons
- Safety tips and guidelines
- Mobile-responsive design

## 🛠️ Technology Stack

- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap
- **Backend**: Flask (Python)
- **ML/Computer Vision**: OpenCV, Deep Learning models
- **Deployment**: Vercel

## 📋 Prerequisites

- Python 3.7+
- Node.js (for Vercel CLI)
- Git
- Webcam access

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/safenest.git
cd safenest
```

2. **Set up a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure ML Models**
The application requires four model files for face and gender detection:
- opencv_face_detector.pbtxt
- opencv_face_detector_uint8.pb
- gender_deploy.prototxt
- gender_net.caffemodel

These will be automatically downloaded from the configured URLs on first run.

## 🏃‍♀️ Running Locally

1. **Start the Flask server**
```bash
python stream_server.py
```

2. **Access the application**
Open your browser and navigate to:
```
http://localhost:3000
```

## 🌐 Deployment

### Deploying to Vercel

1. **Install Vercel CLI**
```bash
npm i -g vercel
```

2. **Deploy**
```bash
vercel
```

3. **Configure Environment Variables (if needed)**
```bash
vercel env add DEBUG
```

## 📁 Project Structure

```
safenest/
├── stream_server.py     # Main Flask application
├── models.py           # ML model management
├── static/            # Static assets
│   ├── css/
│   └── js/
├── models/           # Downloaded ML models (auto-created)
├── requirements.txt  # Python dependencies
├── vercel.json      # Vercel configuration
└── README.md        # Project documentation
```

## 🔒 Security Features

- Real-time monitoring of surroundings
- Instant alerts for suspicious activities
- Camera obstruction detection
- Emergency contact system
- Safety tips and guidelines

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV for computer vision capabilities
- Flask for the web framework
- Bootstrap for UI components
- The open-source community for ML models

## 📧 Contact

Your Name - [arghyadgp4@gmail.com]

Project Link: [https://github.com/braoninja79/safenest-2.0]

## ⚠️ Disclaimer

This application is designed to enhance safety but should not be considered a replacement for proper security measures. Always follow local safety guidelines and contact authorities in case of emergencies.
