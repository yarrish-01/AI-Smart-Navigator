"""
config.py - Central Configuration for AI Smart Navigation Assistant
Contains parameters, thresholds, and risk dictionaries.
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_SAMPLES_DIR = os.path.join(BASE_DIR, "test_samples")

# ==========================================
# 1. VOICE ASSISTANT SETTINGS
# ==========================================
VOICE_SPEECH_RATE = 165       # Words per minute (150-180 is natural and clear)
VOICE_VOLUME = 1.0            # 0.0 to 1.0
ALERT_COOLDOWN_SECONDS = 2.5  # Prevents repeating identical alerts too frequently

# ==========================================
# 2. OBJECT DETECTION & SPATIAL SETTINGS
# ==========================================
YOLO_MODEL_NAME = "yolov8n.pt" # Fast and lightweight for real-time edge processing
YOLO_CONFIDENCE_THRESHOLD = 0.45

# Position detection (horizontal fractions of frame width)
POSITION_LEFT_BOUNDARY = 0.33
POSITION_RIGHT_BOUNDARY = 0.66

# Proximity estimation (bounding box area as percentage of frame area)
# Prototype estimation based on visual perspective
PROXIMITY_FAR_MAX = 0.06       # Less than 6% area = FAR
PROXIMITY_MEDIUM_MAX = 0.22    # 6% to 22% area = MEDIUM
# Greater than 22% area = CLOSE

# Object Risk Classifications
RISK_LEVELS = {
    # High Risk (Fast moving, heavy impact hazard)
    "car": "HIGH",
    "bus": "HIGH",
    "truck": "HIGH",
    "motorcycle": "HIGH",
    "train": "HIGH",

    # Medium Risk (Dynamic or pedestrian collisions)
    "person": "MEDIUM",
    "bicycle": "MEDIUM",
    "dog": "MEDIUM",

    # Low Risk (Static ground obstacles or smaller items)
    "chair": "LOW",
    "couch": "LOW",
    "potted plant": "LOW",
    "bench": "LOW",
    "bottle": "LOW",
    "backpack": "LOW",
    "suitcase": "LOW",
    "fire hydrant": "LOW",
    "stop sign": "LOW"
}

# Priority Engine Weights
RISK_WEIGHTS = {"HIGH": 30, "MEDIUM": 18, "LOW": 8}
PROXIMITY_WEIGHTS = {"CLOSE": 25, "MEDIUM": 12, "FAR": 4}
POSITION_WEIGHTS = {"CENTER": 15, "LEFT": 8, "RIGHT": 8}

# ==========================================
# 3. OCR / TEXT RECOGNITION SETTINGS
# ==========================================
OCR_LANGUAGES = ['en']
OCR_MIN_CONFIDENCE = 0.40       # Minimum detection score to filter out image noise
OCR_MAX_CHARS_READOUT = 200     # Limit spoken characters to avoid endless reading
OCR_MIN_TEXT_LENGTH = 2         # Ignore 1-letter visual artifacts

# Common high-priority signage words for contextual awareness
CRITICAL_SIGNAGE_KEYWORDS = [
    "EXIT", "DANGER", "CAUTION", "WARNING", "STOP",
    "RESTROOM", "WASHROOM", "TOILET", "ROOM", "ENTRANCE",
    "STAIRS", "ELEVATOR", "BUS STOP", "WAY OUT", "PULL", "PUSH"
]
