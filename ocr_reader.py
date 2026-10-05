"""
ocr_reader.py - Optical Character Recognition (OCR) & Voice Reader Module
Extracts text from images or live camera frames using EasyOCR and reads aloud via voice_assistant.
"""

import os
import sys

# Ensure UTF-8 encoding on Windows to prevent UnicodeEncodeError on console progress bars
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
import cv2
import numpy as np
import easyocr
import torch
import config
from voice_assistant import voice_engine

class TextReader:
    def __init__(self, languages=None, min_confidence=config.OCR_MIN_CONFIDENCE):
        self.languages = languages or config.OCR_LANGUAGES
        self.min_confidence = min_confidence

        # Determine GPU availability automatically
        self.use_gpu = torch.cuda.is_available()
        print(f"[OCR Initializing] EasyOCR loading languages={self.languages}, GPU={self.use_gpu}...")
        
        # Initialize reader with verbose=False to avoid Windows charmap encoding errors
        self.reader = easyocr.Reader(self.languages, gpu=self.use_gpu, verbose=False)
        print("[OCR Ready] EasyOCR initialized successfully.")

    def extract_text(self, image_or_path):
        """
        Extracts filtered text blocks and bounding boxes from an image array or file path.
        :param image_or_path: cv2 BGR image numpy array or string filepath
        :return: (extracted_text_summary, detailed_detections, annotated_image)
        """
        if isinstance(image_or_path, str):
            if not os.path.exists(image_or_path):
                raise FileNotFoundError(f"Image not found at {image_or_path}")
            image = cv2.imread(image_or_path)
        else:
            image = image_or_path.copy()

        if image is None:
            return "No image data received.", [], None

        h, w = image.shape[:2]

        # Run EasyOCR
        # Results format: list of (bbox, text, prob)
        raw_results = self.reader.readtext(image)

        valid_detections = []
        for bbox, text, prob in raw_results:
            clean_text = text.strip()
            # Filter low confidence and noisy short single characters
            if prob >= self.min_confidence and len(clean_text) >= config.OCR_MIN_TEXT_LENGTH:
                # Calculate centroid y-coord for vertical sorting (top-to-bottom reading order)
                top_left_y = bbox[0][1]
                top_left_x = bbox[0][0]
                valid_detections.append({
                    "bbox": bbox,
                    "text": clean_text,
                    "confidence": float(prob),
                    "pos_y": top_left_y,
                    "pos_x": top_left_x
                })

        # Sort reading order: Top to Bottom, then Left to Right
        valid_detections.sort(key=lambda d: (d["pos_y"], d["pos_x"]))

        # Build clean spoken summary
        words = [d["text"] for d in valid_detections]
        full_text = " ".join(words)

        # Annotate image for visual feedback
        annotated_img = image.copy()
        for d in valid_detections:
            pts = np.array(d["bbox"], dtype=np.int32)
            cv2.polylines(annotated_img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Label
            label = f"{d['text']} ({int(d['confidence'] * 100)}%)"
            x, y = int(pts[0][0]), int(pts[0][1]) - 8
            if y < 15:
                y = int(pts[0][1]) + 20
            cv2.putText(annotated_img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        return full_text, valid_detections, annotated_img

    def format_speech_message(self, text: str) -> str:
        """Formats the recognized text into an intuitive phrase for the visually impaired user."""
        if not text:
            return "No clear text detected."

        text_upper = text.upper()
        # Check for critical keywords
        for keyword in config.CRITICAL_SIGNAGE_KEYWORDS:
            if keyword in text_upper:
                return f"Notice! Sign says: {text}."

        return f"Reading text: {text}."

    def read_and_speak(self, image_or_path, priority: bool = True):
        """Extracts text and speaks it aloud through the voice assistant."""
        full_text, detections, annotated_img = self.extract_text(image_or_path)
        speech_message = self.format_speech_message(full_text)
        
        print(f"[OCR Result] Extracted: '{full_text}' (Detections: {len(detections)})")
        print(f"[OCR Audio] {speech_message}")
        
        voice_engine.speak(speech_message, priority=priority, force=True)
        return speech_message, full_text, detections, annotated_img


def run_webcam_ocr_mode():
    """
    Interactive live webcam mode:
    Shows live feed. Visually impaired user simulation:
    - Press [SPACEBAR] to capture & read text in front of camera.
    - Press [Q] to exit.
    """
    print("\n" + "="*60)
    print("AI SMART NAVIGATION - OCR CAMERA MODE")
    print("Press [SPACEBAR] to Capture and Read Text")
    print("Press [Q] or [ESC] to Exit")
    print("="*60 + "\n")

    voice_engine.speak("Text reading mode activated. Point your camera at a sign and press space to read.", priority=True)

    reader = TextReader()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[Error] Could not open webcam (camera index 0).")
        voice_engine.speak("Camera could not be opened. Please check webcam connection.", priority=True)
        return

    status_message = "Press [SPACE] to Read Text | [Q] to Quit"
    last_read_text = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Warning] Failed to grab frame.")
            break

        display_frame = frame.copy()
        h, w = display_frame.shape[:2]

        # Draw accessible high-contrast instruction bar at top
        cv2.rectangle(display_frame, (0, 0), (w, 55), (20, 20, 20), -1)
        cv2.putText(display_frame, status_message, (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Draw targeting crosshair / reading area in center
        center_x, center_y = w // 2, h // 2
        box_w, box_h = int(w * 0.75), int(h * 0.55)
        top_left = (center_x - box_w // 2, center_y - box_h // 2)
        bottom_right = (center_x + box_w // 2, center_y + box_h // 2)
        cv2.rectangle(display_frame, top_left, bottom_right, (0, 200, 255), 2)

        if last_read_text:
            cv2.rectangle(display_frame, (0, h - 50), (w, h), (0, 0, 0), -1)
            cv2.putText(display_frame, f"Read: {last_read_text[:50]}", (15, h - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        cv2.imshow("Smart Navigation Assistant - OCR Reader", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 32:  # SPACEBAR pressed
            status_message = "Processing OCR text... Please hold steady."
            voice_engine.speak("Reading text, please hold steady.")
            
            # Process OCR on the captured frame
            msg, full_text, detections, annotated = reader.read_and_speak(frame)
            last_read_text = full_text if full_text else "No text found"
            status_message = f"Read: {last_read_text[:30]} | [SPACE] to Read Again"

            if annotated is not None:
                cv2.imshow("Captured Sign Analysis", annotated)

        elif key in [ord('q'), ord('Q'), 27]:  # 'q' or ESC
            voice_engine.speak("Exiting text reading mode.")
            break

    cap.release()
    cv2.destroyAllWindows()


def run_sample_images_mode():
    """Runs OCR evaluation across all sample images generated in test_samples/."""
    from create_test_images import ensure_test_samples
    print("\n[Step 1] Ensuring sample test images exist...")
    sample_files = ensure_test_samples()

    print("\n[Step 2] Initializing OCR Reader...")
    reader = TextReader()

    print("\n[Step 3] Processing Sample Signage Images...")
    for path in sample_files:
        filename = os.path.basename(path)
        print(f"\n--- Testing Sign: {filename} ---")
        msg, full_text, detections, annotated = reader.read_and_speak(path)
        print(f"Recognized: '{full_text}'")

    print("\nAll sample tests completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Smart Navigation - OCR Reader")
    parser.add_argument("--image", type=str, help="Path to static image file to read")
    parser.add_argument("--webcam", action="store_true", help="Launch live webcam reading mode")
    parser.add_argument("--test-all", action="store_true", help="Run automated test on sample signs")
    args = parser.parse_args()

    if args.image:
        reader = TextReader()
        reader.read_and_speak(args.image)
        import time
        time.sleep(3)
    elif args.webcam:
        run_webcam_ocr_mode()
    elif args.test_all:
        run_sample_images_mode()
        import time
        time.sleep(4)
    else:
        # Default behavior when run directly: test sample images first
        run_sample_images_mode()
        import time
        time.sleep(4)
