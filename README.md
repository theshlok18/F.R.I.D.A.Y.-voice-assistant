# F.R.I.D.A.Y. Voice Assistant 🤖

An advanced AI voice assistant inspired by Tony Stark's F.R.I.D.A.Y. from the Marvel universe. This project combines computer vision, voice recognition, and AI to create an interactive assistant with a futuristic Iron Man-style interface.

## 🚀 Features

- **Real-time Video Processing**: Live camera feed with computer vision capabilities
- **Hand Gesture Recognition**: Draw in the air using finger pinch gestures
- **Voice Commands**: Natural language processing for voice interactions
- **Object Detection**: AI-powered object recognition using CLIP model
- **Futuristic UI**: Iron Man-inspired HUD interface with animations
- **Web Interface**: Flask-based web application with responsive design

## 🛠️ Technologies Used

- **Backend**: Python, Flask
- **Computer Vision**: OpenCV, MediaPipe
- **AI/ML**: PyTorch, CLIP (OpenAI)
- **Frontend**: HTML5, CSS3, JavaScript
- **Voice**: Web Speech API, Speech Synthesis
- **Image Processing**: PIL, NumPy

## 📋 Prerequisites

Before running this application, make sure you have:

- Python 3.7 or higher
- Webcam/Camera access
- Modern web browser (Chrome recommended for voice features)

## 🔧 Installation

1. Clone this repository:
```bash
git clone https://github.com/theshlok18/F.R.I.D.A.Y.-voice-assistant.git
cd F.R.I.D.A.Y.-voice-assistant
```

2. Install required dependencies:
```bash
pip install flask opencv-python mediapipe torch torchvision pillow numpy
pip install git+https://github.com/openai/CLIP.git
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://127.0.0.1:5000
```

## 🎮 Usage

### Voice Commands
- **"Google"** - Opens Google in a new window
- **"Mesh"** or **"Draw"** or **"Start"** - Activates drawing mode
- **"Clear"** - Clears the drawing canvas
- **"Normal"** or **"Stop"** - Returns to normal mode
- **"Structure"** or **"Benzene"** - Opens molecular structure viewer
- **"Scan"** or **"Analyze"** or **"Friday"** - Analyzes current video frame

### Gesture Controls
- **Pinch fingers together** (thumb and index finger) to draw in mesh mode
- **Move while pinching** to create drawings in the air

### Interface Features
- Real-time system stats and logs
- Arc reactor-style voice visualizer
- Scan results with confidence percentages
- Futuristic HUD overlay with corner decorations

## 🏗️ Project Structure

```
├── app.py              # Main Flask application
├── templates/
│   └── index.html      # Frontend interface
└── README.md          # Project documentation
```

## 🎨 Interface Modes

1. **Normal Mode**: Standard video feed with voice commands
2. **Mesh Mode**: Hand tracking with air drawing capabilities
3. **Scan Mode**: Object detection and analysis

## 🔊 Voice Features

- Text-to-speech responses with customizable voice
- Continuous speech recognition
- Real-time subtitle display
- Voice-activated commands

## 🤖 AI Capabilities

- Object detection using OpenAI's CLIP model
- Hand landmark detection with MediaPipe
- Real-time image processing
- Confidence scoring for detections

## 🎯 Future Enhancements

- [ ] Additional voice commands
- [ ] More gesture controls
- [ ] Enhanced object detection
- [ ] Mobile responsiveness
- [ ] Custom wake word detection
- [ ] Integration with smart home devices

## 🐛 Troubleshooting

**Camera not working?**
- Ensure camera permissions are granted
- Check if another application is using the camera

**Voice recognition not working?**
- Use Chrome browser for best compatibility
- Allow microphone permissions when prompted

**Dependencies issues?**
- Make sure all required packages are installed
- Try using a virtual environment

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Credits

**Created by: Shlok**

This project was developed by Shlok as an innovative implementation of an AI voice assistant with computer vision capabilities. The project showcases advanced integration of multiple technologies to create an immersive user experience.

## 🔗 Repository

Original repository: https://github.com/theshlok18/F.R.I.D.A.Y.-voice-assistant

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## ⭐ Show your support

Give a ⭐️ if this project helped you!

---

*"Sometimes you gotta run before you can walk."* - Tony Stark
