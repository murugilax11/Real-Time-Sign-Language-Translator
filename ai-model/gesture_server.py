"""
═══════════════════════════════════════════════════════════
SignSpeak — gesture_server.py (Python AI Module)

What this does:
  1. Runs a simple HTTP server on port 5001
  2. Receives base64 image frames from Node.js
  3. Uses MediaPipe to detect hand landmarks
  4. Uses custom logic to recognize gestures
  5. Returns: { gesture, confidence, landmarks }

Libraries used:
  - Flask: lightweight Python web server (like Express for Python)
  - OpenCV (cv2): image processing — decoding/resizing images
  - MediaPipe: Google's hand landmark detection library
  - NumPy: math and array operations
═══════════════════════════════════════════════════════════
"""

# Standard library imports
import base64      # for decoding base64 images from the browser
import json        # for parsing/returning JSON

# Third-party imports
import cv2                                    # OpenCV for image handling
import mediapipe as mp                        # Google's MediaPipe
import numpy as np                            # numerical operations
from flask import Flask, request, jsonify     # web server
from flask_cors import CORS                   # allow cross-origin requests

# ── FLASK SETUP ──────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # allow Node.js on port 3000 to call this Python server


# ═══════════════════════════════════════════════════════════
# PHASE 5: MEDIAPIPE SETUP
# ─────────────────────────────────────────────────────────
# MediaPipe Hands detects 21 landmarks on each hand.
# Each landmark is an (x, y, z) coordinate:
#   - x and y are normalized (0.0 to 1.0) relative to image size
#   - z is depth (how far from camera — not used in simple gestures)
#
# Landmark Index Reference (0–20):
#   0 = WRIST
#   1–4 = THUMB (1=base, 4=tip)
#   5–8 = INDEX FINGER
#   9–12 = MIDDLE FINGER
#   13–16 = RING FINGER
#   17–20 = PINKY
# ═══════════════════════════════════════════════════════════

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Create the Hands detector
# - max_num_hands: we only look for 1 hand (simpler for beginners)
# - min_detection_confidence: how confident MediaPipe must be to report a hand
# - min_tracking_confidence: how confident to keep tracking between frames
hands_detector = mp_hands.Hands(
    static_image_mode=False,          # False = video mode (uses tracking between frames)
    max_num_hands=1,                  # detect up to 1 hand
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# ═══════════════════════════════════════════════════════════
# GESTURE RECOGNITION LOGIC
# ─────────────────────────────────────────────────────────
# For beginners, we use simple RULES based on finger positions.
# Each finger can be "up" (extended) or "down" (curled).
#
# A more advanced version would use machine learning (a neural
# network trained on thousands of hand images). But rules
# are easier to understand and work well for a small set of gestures.
# ═══════════════════════════════════════════════════════════

def get_finger_states(landmarks):
    """
    Returns a list of 5 booleans: [thumb, index, middle, ring, pinky]
    True = finger is UP (extended)
    False = finger is DOWN (curled)

    How it works:
    - For each finger, we compare the TIP landmark to the MIDDLE landmark.
    - If the tip's y-coordinate is LESS than the middle's y-coordinate,
      the finger is pointing UP (because y increases downward in image coords).
    - Thumb uses x-coordinate comparison (it extends sideways, not up).
    """

    # MediaPipe landmark indices for fingertips and their middle joints
    TIPS   = [4, 8, 12, 16, 20]   # tip of each finger
    MIDDLE = [2, 6, 10, 14, 18]   # middle knuckle of each finger

    finger_up = []

    # THUMB — check horizontal position (x-axis)
    # If thumb tip is to the LEFT of thumb IP joint → thumb is extended
    thumb_tip = landmarks[4]
    thumb_ip  = landmarks[3]
    finger_up.append(thumb_tip.x < thumb_ip.x)

    # OTHER 4 FINGERS — check vertical position (y-axis)
    for tip_idx, mid_idx in zip(TIPS[1:], MIDDLE[1:]):
        tip = landmarks[tip_idx]
        mid = landmarks[mid_idx]
        # If tip is ABOVE the middle joint (smaller y value), finger is up
        finger_up.append(tip.y < mid.y)

    return finger_up  # [thumb, index, middle, ring, pinky]


def classify_gesture(landmarks):
    """
    Recognizes which gesture is being shown.
    Returns (gesture_name, confidence_score).

    Each gesture is defined by which fingers are up or down.
    This is a simple rule-based classifier — perfect for beginners.

    Gesture rules:
      Hello  = all 5 fingers up (open hand)
      Yes    = only thumb up (thumbs up)
      No     = index finger up + shaking (we just detect index up)
      Thanks = flat hand near chin (all 4 fingers up, thumb tucked)
      A      = fist (all fingers down)
      B      = 4 fingers up, thumb across palm
      C      = curved hand shape (partial closure)
    """

    fingers = get_finger_states(landmarks)
    thumb, index, middle, ring, pinky = fingers

    # Count how many fingers are extended
    count = sum(fingers)

    # ── Rule matching ──

    # Hello: all 5 fingers open
    if all(fingers):
        return ("Hello", 0.92)

    # A: closed fist — all fingers down
    if not any(fingers):
        return ("A", 0.88)

    # Yes: only thumb up (thumbs up gesture)
    if thumb and not index and not middle and not ring and not pinky:
        return ("Yes", 0.90)

    # B: 4 fingers up, thumb down
    if not thumb and index and middle and ring and pinky:
        return ("B", 0.85)

    # No: only index finger pointing up
    if not thumb and index and not middle and not ring and not pinky:
        return ("No", 0.82)

    # C: index + thumb up (approximation for the C shape)
    if thumb and index and not middle and not ring and not pinky:
        return ("C", 0.78)

    # Thanks: index + middle + ring up (three-finger salute)
    if not thumb and index and middle and ring and not pinky:
        return ("Thanks", 0.80)

    # If no rule matched, we don't know this gesture
    return ("none", 0.0)


# ═══════════════════════════════════════════════════════════
# FLASK ROUTE — The API endpoint Node.js calls
# ═══════════════════════════════════════════════════════════

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives a base64 JPEG image from Node.js.
    Runs MediaPipe hand detection.
    Returns the detected gesture as JSON.
    """

    # Get the JSON body from Node.js
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    # ── STEP 1: Decode the base64 image ──
    # The browser sent: "data:image/jpeg;base64,/9j/4AAQ..."
    # We need to strip the prefix and decode the raw bytes
    try:
        image_data = data['image']

        # Remove the data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        # Decode base64 string to bytes
        image_bytes = base64.b64decode(image_data)

        # Convert bytes to numpy array (OpenCV needs this format)
        np_array = np.frombuffer(image_bytes, dtype=np.uint8)

        # Decode the numpy array into an actual image (OpenCV format: BGR)
        image_bgr = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if image_bgr is None:
            return jsonify({'error': 'Could not decode image'}), 400

    except Exception as e:
        return jsonify({'error': f'Image decode failed: {str(e)}'}), 400


    # ── STEP 2: Run MediaPipe hand detection ──
    # MediaPipe requires RGB format, but OpenCV uses BGR by default
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Process the image — MediaPipe finds hands and returns landmarks
    results = hands_detector.process(image_rgb)


    # ── STEP 3: Classify the gesture ──
    if not results.multi_hand_landmarks:
        # No hand detected in this frame
        return jsonify({
            'gesture': 'none',
            'confidence': 0.0,
            'landmarks': []
        })

    # Get the first detected hand's landmarks
    hand_landmarks = results.multi_hand_landmarks[0].landmark

    # Run our gesture classifier
    gesture, confidence = classify_gesture(hand_landmarks)

    # ── STEP 4: Format landmarks for frontend visualization ──
    # Convert MediaPipe landmark objects to simple dicts
    # The frontend uses these to draw the hand skeleton on the canvas
    landmarks_list = [
        {'x': lm.x, 'y': lm.y, 'z': lm.z}
        for lm in hand_landmarks
    ]


    # ── STEP 5: Return the result as JSON ──
    return jsonify({
        'gesture': gesture,
        'confidence': round(confidence, 3),
        'landmarks': landmarks_list,
        'fingers': [bool(f) for f in get_finger_states(hand_landmarks)]
    })


# ── HEALTH CHECK ─────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'message': 'Python AI server is running!'})


# ── START THE SERVER ─────────────────────────────────────────
if __name__ == '__main__':
    print("""
╔══════════════════════════════════════╗
║  SignSpeak Python AI — Running ✅     ║
╠══════════════════════════════════════╣
║  URL: http://localhost:5001          ║
║  Endpoint: POST /predict             ║
╚══════════════════════════════════════╝
    """)

    # debug=False in production; True shows detailed error messages
    app.run(host='0.0.0.0', port=5001, debug=True)
