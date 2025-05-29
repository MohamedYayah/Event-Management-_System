import sqlite3
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

DB_PATH = 'events.db'

def get_event_data():
    """Fetch all event records as a DataFrame from the database."""
    print("[DEBUG] Connecting to DB")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('SELECT * FROM events', conn)
    conn.close()
    return df

def train_attendance_model():
    """
    Train a linear regression model to predict attendance.
    Drops rows with missing attendance, date, location, or status.
    Returns (model, feature_columns) or (None, None) if not enough data.
    """
    df = get_event_data()
    # Drop rows with missing data for required columns
    required_cols = ['attendance', 'date', 'location', 'status']
    df = df.dropna(subset=required_cols)
    if 'attendance' not in df.columns or len(df) < 2:
        return None, None
    X = pd.get_dummies(df[['date', 'location', 'status']], drop_first=True)
    y = df['attendance']
    # Check for NaNs
    if X.isnull().values.any() or y.isnull().values.any():
        return None, None
    model = LinearRegression()
    model.fit(X, y)
    return model, X.columns

def get_feature_importances(model, feature_columns):
    """
    Returns a sorted list of (feature, importance) tuples for the linear regression model.
    Importance is the absolute value of the coefficient.
    """
    if hasattr(model, 'coef_'):
        importances = list(zip(feature_columns, model.coef_))
        # Sort by absolute value, descending
        importances.sort(key=lambda x: abs(x[1]), reverse=True)
        return importances
    return []

def predict_attendance_for_event(event_id):
    print(f"[DEBUG] predict_attendance_for_event called with event_id={event_id}")
    try:
        model, feature_columns = train_attendance_model()
        print("[DEBUG] Loaded model and feature columns")
        if model is None:
            print("[ERROR] Attendance prediction model not found.")
            return []
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        print(f"[DEBUG] Fetching attendees and event data for event_id={event_id}")
        cursor.execute("SELECT id, name FROM attendees WHERE event_id=?", (event_id,))
        attendees = cursor.fetchall()
        cursor.execute("SELECT date, location, status FROM events WHERE id=?", (event_id,))
        event_data = cursor.fetchone()
        conn.close()
        if event_data is None or not attendees:
            print(f"[DEBUG] No event data or attendees found for event_id={event_id}")
            return []
        event_features = {
            'date': event_data[0],
            'location': event_data[1],
            'status': event_data[2]
        }
        print(f"[DEBUG] Predicting attendance for {len(attendees)} attendees")
        # For demo: predict same status for all attendees (since model is event-level)
        predicted_status = predict_attendance(event_features, model, feature_columns)
        print(f"[DEBUG] Predicted status: {predicted_status}")
        # Return a list of (attendee_id, predicted_status)
        return [(att[0], predicted_status) for att in attendees]
    except Exception as e:
        print(f"[ERROR] Exception in predict_attendance_for_event: {e}")
        return []

def predict_attendance(event_features, model, feature_columns):
    """
    Predict attendance for a single event using the trained model.
    event_features: dict with keys matching model features.
    model: trained LinearRegression model.
    feature_columns: columns used in training.
    Returns integer prediction or None.
    """
    print("[DEBUG] Predicting attendance for event features={event_features}")
    try:
        X_pred = pd.DataFrame([event_features])
        X_pred = pd.get_dummies(X_pred)
        # Ensure all columns match
        for col in feature_columns:
            if col not in X_pred.columns:
                X_pred[col] = 0
        X_pred = X_pred[feature_columns]
        pred = model.predict(X_pred)
        return int(round(pred[0]))
    except Exception as e:
        print(f"[ERROR] Exception in predict_attendance: {e}")
        return None
