import sqlite3
import cv2
import mediapipe as mp
import numpy as np
import json
import os
import logging
from tqdm import tqdm

def migrate_attendee_landmarks(db_path, photo_dir):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    attendees = cursor.execute('SELECT id, face_landmarks, name, email FROM attendees').fetchall()
    mp_face_mesh = mp.solutions.face_mesh
    updated = 0
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        for attendee in tqdm(attendees, desc='Migrating attendees'):
            try:
                # Skip if already migrated
                try:
                    landmarks = json.loads(attendee['face_landmarks'])
                    if isinstance(landmarks, list) and isinstance(landmarks[0], list) and len(landmarks[0]) == 3:
                        continue  # Already migrated
                except Exception:
                    pass
                # Try to find photo file by attendee id, email, or name
                photo_path = None
                for ext in ['.jpg', '.jpeg', '.png']:
                    for key in [str(attendee['id']), attendee['email'], attendee['name']]:
                        candidate = os.path.join(photo_dir, f"{key}{ext}")
                        if os.path.exists(candidate):
                            photo_path = candidate
                            break
                    if photo_path:
                        break
                if not photo_path:
                    logging.warning(f"No photo found for attendee {attendee['id']} ({attendee['name']})")
                    continue
                img = cv2.imread(photo_path)
                if img is None:
                    logging.warning(f"Could not read image for attendee {attendee['id']} from {photo_path}")
                    continue
                results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if results.multi_face_landmarks:
                    face_landmarks = json.dumps([[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark])
                    cursor.execute('UPDATE attendees SET face_landmarks = ? WHERE id = ?', (face_landmarks, attendee['id']))
                    updated += 1
                else:
                    logging.warning(f"No face detected for attendee {attendee['id']} ({attendee['name']})")
            except Exception as e:
                logging.error(f"Error processing attendee {attendee['id']} ({attendee['name']}): {e}")
    conn.commit()
    conn.close()
    print(f"Migration complete. Updated {updated} attendees.")

if __name__ == '__main__':
    # Adjust paths as needed
    db_path = os.path.join(os.path.dirname(__file__), 'events.db')
    photo_dir = os.path.join(os.path.dirname(__file__), 'attendee_photos')
    migrate_attendee_landmarks(db_path, photo_dir)
