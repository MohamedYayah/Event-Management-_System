
import joblib
import os

def load_model():
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


import numpy as np
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'attendance_model.pkl')
DB_PATH = os.path.join(os.path.dirname(__file__), 'events.db')

def extract_ml_data():
    """
    Extracts features (X) and labels (y) from attendees/events tables for ML.
    Features: day_of_week, hour, is_weekend, location, event_status, event_attendance, event_type (if available), attendee_role (if available), previous_attendance_rate (if available)
    Label: 1 if attendee.status == 'Present', else 0
    Returns: X, y (numpy arrays)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT a.id, a.status, a.role, a.previous_attendance_rate, e.date, e.time, e.location, e.status, e.attendance, e.type
            FROM attendees a
            JOIN events e ON a.event_id = e.id
        ''')
        rows = cursor.fetchall()
    except Exception:
        # Fallback if some columns are missing
        cursor.execute('''
            SELECT a.id, a.status, NULL as role, NULL as previous_attendance_rate, e.date, e.time, e.location, e.status, e.attendance, NULL as type
            FROM attendees a
            JOIN events e ON a.event_id = e.id
        ''')
        rows = cursor.fetchall()
    conn.close()
    X = []
    y = []
    locations = set()
    event_statuses = set()
    event_types = set()
    attendee_roles = set()
    for row in rows:
        att_id, att_status, att_role, att_prev_rate, date_str, time_str, location, event_status, event_attendance, event_type = row
        # Parse date/time
        try:
            day_of_week = datetime.strptime(date_str, '%Y-%m-%d').weekday()  # 0=Monday
        except Exception:
            day_of_week = 0
        is_weekend = 1 if day_of_week >= 5 else 0
        try:
            hour = int(time_str.split(':')[0])
        except Exception:
            hour = 0
        locations.add(location)
        event_statuses.add(event_status)
        event_types.add(event_type)
        attendee_roles.add(att_role)
        X.append([
            day_of_week, hour, is_weekend, location, event_status, event_attendance or 0,
            event_type, att_role, att_prev_rate if att_prev_rate is not None else 0.0
        ])
        y.append(1 if att_status and att_status.lower() == 'present' else 0)
    # One-hot encode location, event_status, event_type, attendee_role
    if not X:
        return np.empty((0,9)), np.empty((0,))
    X_np = np.array(X, dtype=object)
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cols_to_encode = X_np[:,[3,4,6,7]]  # location, event_status, event_type, attendee_role
    enc.fit(cols_to_encode)
    encoded = enc.transform(cols_to_encode)
    X_final = np.concatenate([
        X_np[:,[0,1,2,5,8]].astype(float),  # day_of_week, hour, is_weekend, event_attendance, prev_attendance_rate
        encoded
    ], axis=1)
    return X_final, np.array(y)

def predict_attendance_for_event(event_id):
    """
    Predict attendance for a future event. Returns [(attendee_id, predicted_status)]
    """
    from sklearn.preprocessing import OneHotEncoder
    model = load_model()
    if model is None:
        raise RuntimeError("Attendance prediction model not found. Please retrain the model first from the Admin Dashboard.")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT a.id, a.role, a.previous_attendance_rate, e.date, e.time, e.location, e.status, e.attendance, e.type
            FROM attendees a
            JOIN events e ON a.event_id = e.id
            WHERE e.id = ?
        ''', (event_id,))
        rows = cursor.fetchall()
    except Exception:
        cursor.execute('''
            SELECT a.id, NULL as role, NULL as previous_attendance_rate, e.date, e.time, e.location, e.status, e.attendance, NULL as type
            FROM attendees a
            JOIN events e ON a.event_id = e.id
            WHERE e.id = ?
        ''', (event_id,))
        rows = cursor.fetchall()
    conn.close()
    if not rows:
        return []
    X = []
    att_ids = []
    for row in rows:
        att_id, att_role, att_prev_rate, date_str, time_str, location, event_status, event_attendance, event_type = row
        try:
            day_of_week = datetime.strptime(date_str, '%Y-%m-%d').weekday()
        except Exception:
            day_of_week = 0
        is_weekend = 1 if day_of_week >= 5 else 0
        try:
            hour = int(time_str.split(':')[0])
        except Exception:
            hour = 0
        X.append([
            day_of_week, hour, is_weekend, location, event_status, event_attendance or 0,
            event_type, att_role, att_prev_rate if att_prev_rate is not None else 0.0
        ])
        att_ids.append(att_id)
    # One-hot encode as in training
    X_np = np.array(X, dtype=object)
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    cols_to_encode = X_np[:,[3,4,6,7]]
    enc.fit(cols_to_encode)
    encoded = enc.transform(cols_to_encode)
    X_final = np.concatenate([
        X_np[:,[0,1,2,5,8]].astype(float),
        encoded
    ], axis=1)
    y_pred = model.predict(X_final)
    return list(zip(att_ids, y_pred))

    """
    Extracts features (X) and labels (y) from attendees/events tables for ML.
    Features: day_of_week, hour, location, event_status, event_attendance
    Label: 1 if attendee.status == 'Present', else 0
    Returns: X, y (numpy arrays)
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT a.status, e.date, e.time, e.location, e.status, e.attendance
        FROM attendees a
        JOIN events e ON a.event_id = e.id
    ''')
    rows = cursor.fetchall()
    conn.close()
    X = []
    y = []
    locations = set()
    event_statuses = set()
    for row in rows:
        att_status, date_str, time_str, location, event_status, event_attendance = row
        # Parse date/time
        try:
            day_of_week = datetime.strptime(date_str, '%Y-%m-%d').weekday()  # 0=Monday
        except Exception:
            day_of_week = 0
        try:
            hour = int(time_str.split(':')[0])
        except Exception:
            hour = 0
        locations.add(location)
        event_statuses.add(event_status)
        X.append([day_of_week, hour, location, event_status, event_attendance or 0])
        y.append(1 if att_status and att_status.lower() == 'present' else 0)
    # One-hot encode location and event_status
    if not X:
        return np.empty((0,5)), np.empty((0,))
    X_np = np.array(X, dtype=object)
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    loc_status = X_np[:,[2,3]]
    enc.fit(loc_status)
    loc_status_encoded = enc.transform(loc_status)
    X_final = np.concatenate([
        X_np[:,[0,1,4]].astype(float),  # day_of_week, hour, event_attendance
        loc_status_encoded
    ], axis=1)
    return X_final, np.array(y)

def train_and_evaluate_model(X_train, y_train, X_test, y_test, model=None):
    """
    Train a model, evaluate, and save it. Returns accuracy and classification report dict.
    """
    if model is None:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    joblib.dump(model, MODEL_PATH)
    return acc, report

def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None
