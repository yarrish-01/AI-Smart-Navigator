"""
vision_detector.py - Real-Time YOLO Object Detection & Spatial Priority Engine
Features:
- Pretrained YOLOv8 detection
- Filter for relevant navigation obstacles (vehicles, pedestrians, path obstacles)
- Horizontal position detection (LEFT / CENTER / RIGHT)
- Proximity estimation via bounding box screen area (CLOSE / MEDIUM / FAR)
- Hierarchical risk assessment (HIGH / MEDIUM / LOW)
- Priority scoring engine to select the single most critical obstacle to announce
- Temporal stability and cooldown suppression to avoid voice spamming
"""

import time
import cv2
import numpy as np
from ultralytics import YOLO
import config
from voice_assistant import voice_engine


class SpatialVisionDetector:
    def __init__(self, model_name=config.YOLO_MODEL_NAME, conf_thresh=config.YOLO_CONFIDENCE_THRESHOLD):
        self.conf_thresh = conf_thresh
        print(f"[Vision Initializing] Loading YOLO model '{model_name}'...")
        # Pretrained YOLO model (auto-downloads on first run if not cached)
        self.model = YOLO(model_name)
        print("[Vision Ready] YOLO model loaded.")

        # Temporal stability tracking: tracks detection count over consecutive frames
        # {class_name: consecutive_frame_count}
        self.tracking_history = {}
        self.min_consecutive_frames = 2  # Must appear in at least 2 frames to trigger speech
        self.last_spoken_alert = ""
        self.last_spoken_time = 0.0

    def compute_position(self, bbox, frame_width):
        """
        Determines horizontal position of the object: LEFT, CENTER, or RIGHT.
        :param bbox: [x1, y1, x2, y2]
        :param frame_width: width of camera frame
        :return: "LEFT", "CENTER", or "RIGHT"
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        fraction = center_x / float(frame_width)

        if fraction < config.POSITION_LEFT_BOUNDARY:
            return "LEFT"
        elif fraction > config.POSITION_RIGHT_BOUNDARY:
            return "RIGHT"
        else:
            return "CENTER"

    def estimate_proximity(self, bbox, frame_width, frame_height):
        """
        Estimates proximity based on bounding-box area relative to total frame area.
        Prototype estimation (NOT an exact metric measurement).
        :param bbox: [x1, y1, x2, y2]
        :return: "FAR", "MEDIUM", or "CLOSE"
        """
        x1, y1, x2, y2 = bbox
        box_area = max(0, x2 - x1) * max(0, y2 - y1)
        total_frame_area = frame_width * frame_height
        area_ratio = box_area / float(total_frame_area)

        if area_ratio < config.PROXIMITY_FAR_MAX:
            return "FAR", area_ratio
        elif area_ratio < config.PROXIMITY_MEDIUM_MAX:
            return "MEDIUM", area_ratio
        else:
            return "CLOSE", area_ratio

    def calculate_priority_score(self, risk_level, proximity, position):
        """
        Calculates priority score combining risk, proximity, and position.
        Score = Risk_Weight + Proximity_Weight + Position_Weight
        """
        score = 0
        score += config.RISK_WEIGHTS.get(risk_level, 5)
        score += config.PROXIMITY_WEIGHTS.get(proximity, 5)
        score += config.POSITION_WEIGHTS.get(position, 5)
        return score

    def format_alert_phrase(self, class_name, risk_level, proximity, position):
        """
        Generates natural spoken alert string.
        Examples:
        - HIGH + CLOSE: "Warning! Car close ahead."
        - MEDIUM + LEFT: "Person on your left."
        - LOW + CENTER: "Chair ahead."
        """
        # Position wording
        pos_word = "ahead" if position == "CENTER" else f"on your {position.lower()}"

        if risk_level == "HIGH":
            if proximity == "CLOSE":
                return f"Warning! {class_name.capitalize()} close {pos_word}."
            elif proximity == "MEDIUM":
                return f"Caution, {class_name} approaching {pos_word}."
            else:
                return f"{class_name.capitalize()} {pos_word}."
        elif risk_level == "MEDIUM":
            if proximity == "CLOSE":
                return f"Caution, {class_name} close {pos_word}."
            else:
                return f"{class_name.capitalize()} {pos_word}."
        else:
            # Low risk obstacles (chairs, bags, etc.)
            if proximity == "CLOSE":
                return f"Obstacle, {class_name} directly {pos_word}."
            elif proximity == "MEDIUM":
                return f"{class_name.capitalize()} {pos_word}."
            else:
                # Do not spam low-risk far obstacles
                return None

    def process_frame(self, frame):
        """
        Runs YOLO detection, filtering, proximity, position, and priority engine on a single frame.
        :param frame: cv2 BGR frame
        :return: (annotated_frame, candidate_objects, top_priority_object, alert_message)
        """
        h, w = frame.shape[:2]
        results = self.model(frame, conf=self.conf_thresh, verbose=False)

        candidate_objects = []
        current_frame_classes = set()

        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id].lower()
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()

                # Filter only relevant navigation objects defined in RISK_LEVELS
                if cls_name not in config.RISK_LEVELS:
                    continue

                current_frame_classes.add(cls_name)

                # Temporal stability check
                self.tracking_history[cls_name] = self.tracking_history.get(cls_name, 0) + 1

                risk_level = config.RISK_LEVELS.get(cls_name, "LOW")
                position = self.compute_position(xyxy, w)
                proximity, area_ratio = self.estimate_proximity(xyxy, w, h)
                prio_score = self.calculate_priority_score(risk_level, proximity, position)

                candidate_objects.append({
                    "class_name": cls_name,
                    "confidence": conf,
                    "bbox": xyxy,
                    "risk": risk_level,
                    "position": position,
                    "proximity": proximity,
                    "area_ratio": area_ratio,
                    "priority_score": prio_score,
                    "is_stable": self.tracking_history[cls_name] >= self.min_consecutive_frames
                })

        # Decay tracking history for objects no longer visible
        for tracked_cls in list(self.tracking_history.keys()):
            if tracked_cls not in current_frame_classes:
                self.tracking_history[tracked_cls] = max(0, self.tracking_history[tracked_cls] - 1)

        # Sort candidate objects by Priority Score in descending order
        candidate_objects.sort(key=lambda o: o["priority_score"], reverse=True)

        top_priority_object = None
        alert_message = None

        # Pick highest priority stable object
        for obj in candidate_objects:
            if obj["is_stable"]:
                top_priority_object = obj
                alert_message = self.format_alert_phrase(
                    obj["class_name"],
                    obj["risk"],
                    obj["proximity"],
                    obj["position"]
                )
                break

        # Draw visual annotations on frame for developer / presenter view
        annotated_frame = frame.copy()
        
        # Draw vertical grid lines for Left / Center / Right zones
        left_x = int(w * config.POSITION_LEFT_BOUNDARY)
        right_x = int(w * config.POSITION_RIGHT_BOUNDARY)
        cv2.line(annotated_frame, (left_x, 0), (left_x, h), (100, 100, 100), 1, cv2.LINE_AA)
        cv2.line(annotated_frame, (right_x, 0), (right_x, h), (100, 100, 100), 1, cv2.LINE_AA)
        
        cv2.putText(annotated_frame, "LEFT", (20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(annotated_frame, "CENTER / AHEAD", (left_x + 20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(annotated_frame, "RIGHT", (right_x + 20, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        for obj in candidate_objects:
            x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
            risk = obj["risk"]

            # Color coding: Red = High, Amber = Medium, Green = Low
            if risk == "HIGH":
                color = (0, 0, 255)
            elif risk == "MEDIUM":
                color = (0, 165, 255)
            else:
                color = (0, 255, 0)

            # Highlight top priority object with thicker outline
            thickness = 3 if (top_priority_object and obj == top_priority_object) else 1
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

            label = f"{obj['class_name']} | {obj['risk']} | {obj['proximity']} | {obj['position']}"
            cv2.putText(annotated_frame, label, (x1, max(18, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Trigger voice alert if valid and not on cooldown
        if alert_message:
            current_time = time.time()
            is_high_risk = top_priority_object and top_priority_object["risk"] == "HIGH"
            
            # Announce via voice assistant
            voice_engine.speak(alert_message, priority=is_high_risk)
            self.last_spoken_alert = alert_message
            self.last_spoken_time = current_time

        return annotated_frame, candidate_objects, top_priority_object, alert_message


def run_vision_webcam():
    """Runs live webcam obstacle detection with voice alerts."""
    print("\n" + "="*60)
    print("AI SMART NAVIGATION - SPATIAL OBSTACLE DETECTION")
    print("Press [Q] or [ESC] to Exit")
    print("="*60 + "\n")

    detector = SpatialVisionDetector()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[Error] Could not open webcam.")
        voice_engine.speak("Webcam not detected.", priority=True)
        return

    voice_engine.speak("Obstacle detection active.", priority=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, candidates, top_obj, alert_msg = detector.process_frame(frame)

        # Top overlay banner
        h, w = frame.shape[:2]
        cv2.rectangle(annotated_frame, (0, 0), (w, 45), (15, 15, 15), -1)
        status_text = f"Top Alert: {alert_msg}" if alert_msg else "Scanning surroundings..."
        cv2.putText(annotated_frame, status_text, (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        cv2.imshow("Smart Navigation - Obstacle Detector", annotated_frame)
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('Q'), 27]:
            voice_engine.speak("Exiting obstacle detection.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_vision_webcam()
