import cv2
import threading
import time
import numpy as np
import mediapipe as mp
import math
from flask import Flask, render_template, Response, jsonify, request
from PIL import Image

# --- 1. SYSTEM INIT ---
app = Flask(__name__)

# --- CONFIGURATION ---
SYSTEM_NAME = "FRIDAY"
CURRENT_MODE = "normal" 

# --- MEDIAPIPE (OPTIMIZED HANDS) ---
mp_hands = mp.solutions.hands
try:
    hands_detector = mp_hands.Hands(
        max_num_hands=2,
        model_complexity=0, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
except:
    hands_detector = None
    print(">> ERROR: MediaPipe not installed correctly.")

outputFrame = None
lock = threading.Lock()
cap = cv2.VideoCapture(0)

drawing_canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
prev_x, prev_y = 0, 0
is_drawing = False

model = None
try:
    import torch
    import clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    LABELS = ["cat", "dog", "laptop", "Gun", "coffee cup", "person", "keyboard"]
    text_inputs = clip.tokenize(LABELS).to(device)
except:
    model = None # Fallback

def calculate_distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def process_hands(image):
    global drawing_canvas, prev_x, prev_y, is_drawing
    if not hands_detector: return image
    
    image.flags.writeable = False
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(rgb)
    image.flags.writeable = True
    
    height, width, _ = image.shape

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            idx_x = int(hand_landmarks.landmark[8].x * width)
            idx_y = int(hand_landmarks.landmark[8].y * height)
            thm_x = int(hand_landmarks.landmark[4].x * width)
            thm_y = int(hand_landmarks.landmark[4].y * height)

            cv2.circle(image, (idx_x, idx_y), 10, (0, 255, 255), 2) 
            cv2.circle(image, (thm_x, thm_y), 10, (255, 0, 255), 2) 

            #  Logic
            dist = calculate_distance((idx_x, idx_y), (thm_x, thm_y))
            if dist < 40:
                cv2.line(image, (idx_x, idx_y), (thm_x, thm_y), (0, 255, 0), 3) 
                if prev_x == 0 and prev_y == 0: prev_x, prev_y = idx_x, idx_y
                cv2.line(drawing_canvas, (prev_x, prev_y), (idx_x, idx_y), (0, 255, 255), 5)
                prev_x, prev_y = idx_x, idx_y
                is_drawing = True
            else:
                prev_x, prev_y = 0, 0
                is_drawing = False

    gray = cv2.cvtColor(drawing_canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    img_bg = cv2.bitwise_and(image, image, mask=mask_inv)
    img_fg = cv2.bitwise_and(drawing_canvas, drawing_canvas, mask=mask)
    return cv2.add(img_bg, img_fg)

def capture_frames():
    global outputFrame, cap, CURRENT_MODE
    cap.set(3, 1280)
    cap.set(4, 720)
    
    while True:
        success, frame = cap.read()
        if not success:
            time.sleep(0.1)
            continue

        frame = cv2.resize(frame, (1280, 720))
        frame = cv2.flip(frame, 1)

        if CURRENT_MODE == "mesh":
            frame = process_hands(frame)
        
        cv2.putText(frame, f"SYSTEM: {CURRENT_MODE.upper()}", (30, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 240, 255), 2)

        with lock:
            outputFrame = frame.copy()

t = threading.Thread(target=capture_frames)
t.daemon = True
t.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def gen():
        global outputFrame
        while True:
            with lock:
                if outputFrame is None: continue
                _, encoded = cv2.imencode(".jpg", outputFrame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + bytearray(encoded) + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/command', methods=['POST'])
def command_handler():
    global CURRENT_MODE, drawing_canvas
    data = request.json
    text = data.get('command', '').lower()
    
    response = {"action": "none", "speech": ""}

    if "google" in text:
        response["action"] = "open_window"
        response["url"] = "https://www.google.com"
        response["speech"] = "Accessing Google Database."

    elif "mesh" in text or "draw" in text or "start" in text:
        CURRENT_MODE = "mesh"
        response["action"] = "mode_switch"
        response["speech"] = "Creative mode active. Pinch fingers to draw."
    
    elif "clear" in text:
        drawing_canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        response["speech"] = "Canvas wiped."

    elif "normal" in text or "stop" in text:
        CURRENT_MODE = "normal"
        response["action"] = "mode_switch"
        response["speech"] = "Disengaging UI."
    elif "structure" in text or "benzene" in text:
        response["action"] = "open_window"
        response["url"] = "https://app.molview.com/"
        response["speech"] = "this is a 3d structure of benzene"

    elif "scan" in text or "analyze" in text or "friday" in text:
        response["action"] = "scan"
        response["speech"] = "Scanning environment."

    return jsonify(response)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    global outputFrame
    with lock:
        if outputFrame is None: 
            return jsonify({"ui_text": "VIDEO LOSS", "speech_text": "Camera offline.", "is_threat": False})
        frame_rgb = cv2.cvtColor(outputFrame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

    label = "analyzing..."
    confidence = 0.0

    if model:
        try:
            image_input = preprocess(pil_image).unsqueeze(0).to(device)
            with torch.no_grad():
                logits_per_image, _ = model(image_input, text_inputs)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            best_idx = probs.argmax()
            confidence = float(probs[best_idx])
            label = LABELS[best_idx]
            if confidence < 0.2: label = "Unknown Object"
        except:
            label = "Error"
    else:
        import random
        sim_labels = ["Human", "Smartphone", "Workstation", "Coffee"]
        label = random.choice(sim_labels)
        confidence = 0.88

    return jsonify({
        "ui_text": f"DETECTED: {label.upper()}",

        "speech_text": f"I see a {label} sir, you are in danger!,deploying saftey protocols .",
        "confidence": int(confidence * 100),
        "is_threat": (label == "danger weapon ")
    })


if __name__ == '__main__':
    app.run()
    