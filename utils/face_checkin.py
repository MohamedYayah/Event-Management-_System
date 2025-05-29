import cv2
import mediapipe as mp
import sqlite3
from datetime import datetime
import numpy as np

DB_PATH = 'event_management.db'

def log_attendance(username, method, event_id):
    """
    Log check-in attendance for the user with timestamp and method (face/motion) for a specific event.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Add timestamp column if not exists
    try:
        cursor.execute("ALTER TABLE attendees ADD COLUMN timestamp TEXT")
    except sqlite3.OperationalError:
        pass  # Already exists
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("""
        INSERT INTO attendees (event_id, name, status, role, previous_attendance_rate, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (event_id, username, f"checked_in_{method}", 'attendee', 1.0, now))
    conn.commit()
    conn.close()


def detect_face_via_webcam(timeout=10, min_detected_duration=2):
    """
    Opens webcam, detects a face using MediaPipe, returns True if face detected for at least min_detected_duration seconds within timeout seconds.
    Strengthened: Higher confidence, continuous detection, robust error handling, user feedback.
    """
    mp_face = mp.solutions.face_detection
    mp_draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return False
    detected = False
    detected_start = None
    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.85) as face_detection:
        start_time = cv2.getTickCount()
        freq = cv2.getTickFrequency()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cv2.putText(frame, 'Webcam Error!', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
                cv2.imshow('Face Check-In', frame)
                cv2.waitKey(1000)
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb)
            current_time = (cv2.getTickCount() - start_time) / freq
            if results.detections:
                for detection in results.detections:
                    mp_draw.draw_detection(frame, detection)
                if detected_start is None:
                    detected_start = cv2.getTickCount()
                elapsed_detected = (cv2.getTickCount() - detected_start) / freq
                if elapsed_detected >= min_detected_duration:
                    detected = True
                    cv2.putText(frame, 'Face detected! Hold still. Press Q to confirm.', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                else:
                    cv2.putText(frame, f'Hold still for {min_detected_duration - elapsed_detected:.1f}s', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
            else:
                detected_start = None
                cv2.putText(frame, 'No face detected', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Check-In', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if current_time > timeout:
                break
    cap.release()
    cv2.destroyAllWindows()
    if detected:
        print("Face check-in successful!")
    else:
        print("Face check-in failed. Make sure your face is clearly visible and well-lit.")
    return detected



def detect_motion_via_webcam(timeout=10, min_area=5000):
    """
    Opens webcam, detects motion. Returns True if significant motion detected within timeout seconds.
    """
    cap = cv2.VideoCapture(0)
    detected = False
    first_frame = None
    start_time = cv2.getTickCount()
    freq = cv2.getTickFrequency()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if first_frame is None:
            first_frame = gray
            continue
        frame_delta = cv2.absdiff(first_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) > min_area:
                detected = True
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        status_text = 'Motion detected! Press Q to confirm.' if detected else 'No motion detected.'
        color = (0,255,0) if detected else (0,0,255)
        cv2.putText(frame, status_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow('Motion Check-In', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elapsed = (cv2.getTickCount() - start_time) / freq
        if elapsed > timeout:
            break
    cap.release()
    cv2.destroyAllWindows()
    return detected

if __name__ == "__main__":
    username = input("Enter username for demo logging: ")
    method = input("Type 'face' or 'motion': ").strip()
    if method == 'face':
        result = detect_face_via_webcam()
    else:
        result = detect_motion_via_webcam()
    if result:
        log_attendance(username, method)
        print("Check-in successful!")
    else:
        print("Check-in failed.")
