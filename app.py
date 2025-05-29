import os
from dotenv import load_dotenv
import logging
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

# Setup logging
logging.basicConfig(
    filename=os.path.join(os.path.dirname(__file__), 'app.log'),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s'
)

print("PATH at Flask startup:", os.environ["PATH"])
try:
    import mediapipe as mp
    print("mediapipe imported successfully at Flask startup!")
except Exception as e:
    logging.error(f"mediapipe import failed at Flask startup: {e}")
from flask import Flask, render_template, request, redirect, url_for, flash, session, make_response, jsonify
from flask_mail import Mail, Message
import requests
import io
import csv
from ml import ml_utils
from forms import LoginForm
import sqlite3
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from utils.event_validation import validate_event_form


# Load environment variables from .env
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY')

# Flask-Mail config from environment
app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.environ.get('MAIL_USE_TLS', 'True') == 'True'
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD')
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER')
mail = Mail(app)
DATABASE = os.path.join(os.path.dirname(__file__), 'events.db')

# SQLAlchemy setup
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DATABASE}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- Profile Model ---
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Profile(db.Model):
    __tablename__ = 'profiles'
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), primary_key=True)
    profile_picture = db.Column(db.Text)
    name = db.Column(db.Text)
    email = db.Column(db.Text)
    phone = db.Column(db.Text)
    address = db.Column(db.Text)
    gender = db.Column(db.Text)
    city = db.Column(db.Text)
    user = db.relationship('User', backref=db.backref('profile', uselist=False))

db.init_app(app)
migrate = Migrate(app, db)

class User(db.Model, UserMixin):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    notification_enabled = db.Column(db.Boolean, default=True)
    theme = db.Column(db.String(20), default='light')

    @staticmethod
    def get_by_email(email):
        """Load user by email using SQLAlchemy ORM."""
        return User.query.filter_by(email=email).first()

    @staticmethod
    def get(user_id):
        """Load user by user_id using SQLAlchemy ORM."""
        return User.query.get(int(user_id))

class LoginActivity(db.Model):
    __tablename__ = 'login_activity'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    timestamp = db.Column(db.DateTime, server_default=db.func.now())
    ip_address = db.Column(db.String(45))


class Event(db.Model):
    __tablename__ = 'events'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    date = db.Column(db.String(20))
    time = db.Column(db.String(20))
    location = db.Column(db.String(200))
    status = db.Column(db.String(50))
    description = db.Column(db.Text)
    attendance = db.Column(db.Integer)

class Attendee(db.Model):
    __tablename__ = 'attendees'
    id = db.Column(db.Integer, primary_key=True)
    event_id = db.Column(db.Integer, db.ForeignKey('events.id'))
    name = db.Column(db.String(100))
    email = db.Column(db.String(120))
    status = db.Column(db.String(50))
    face_landmarks = db.Column(db.Text)
    role = db.Column(db.String(50))
    previous_attendance_rate = db.Column(db.Float)

class Notification(db.Model):
    __tablename__ = 'notifications'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    message = db.Column(db.Text, nullable=False)
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())


# Add current_year to all templates
def inject_current_year():
    from datetime import datetime
    return dict(current_year=datetime.now().year)
app.context_processor(inject_current_year)


# --- Flask-Login Setup ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    """Flask-Login user loader callback."""
    return db.session.get(User, int(user_id))

def get_db_connection():
    """Create and return a new database connection (rows as dicts)."""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/notifications')
@login_required
def notifications():
    conn = get_db_connection()
    notes = conn.execute('SELECT id, message, is_read, created_at FROM notifications WHERE user_id = ? ORDER BY created_at DESC', (current_user.id,)).fetchall()
    conn.close()
    return render_template('notifications.html', notifications=notes)


@app.route('/dashboard')
@login_required
def dashboard():
    if hasattr(current_user, 'is_admin') and current_user.is_admin:
        return redirect(url_for('admin_dashboard'))

    profile = Profile.query.filter_by(user_id=current_user.id).first()

    conn = get_db_connection()
    stats = {
        'upcoming': conn.execute("SELECT COUNT(*) FROM events WHERE status = 'Upcoming'").fetchone()[0],
        'today': conn.execute("SELECT COUNT(*) FROM events WHERE date = date('now')").fetchone()[0],
        'completed': conn.execute("SELECT COUNT(*) FROM events WHERE status = 'Completed'").fetchone()[0],
        'total': conn.execute("SELECT COUNT(*) FROM events").fetchone()[0],
        'cancelled': conn.execute("SELECT COUNT(*) FROM events WHERE status = 'Cancelled'").fetchone()[0]
    }
    q = request.args.get('q', '').strip()
    date = request.args.get('date', '').strip()
    location = request.args.get('location', '').strip()
    status = request.args.get('status', '').strip()
    query = 'SELECT * FROM events WHERE 1=1'
    params = []
    if q:
        query += ' AND LOWER(title) LIKE ?'
        params.append(f'%{q.lower()}%')
    if date:
        query += ' AND date = ?'
        params.append(date)
    if location:
        query += ' AND LOWER(location) LIKE ?'
        params.append(f'%{location.lower()}%')
    if status:
        query += ' AND status = ?'
        params.append(status)
    sort_by = request.args.get('sort_by', 'date')
    sort_order = request.args.get('sort_order', 'asc')
    valid_sort_by = ['date', 'title', 'location', 'status']
    valid_sort_order = ['asc', 'desc']
    if sort_by not in valid_sort_by:
        sort_by = 'date'
    if sort_order not in valid_sort_order:
        sort_order = 'asc'
    query += f' ORDER BY {sort_by} {sort_order.upper()}'
    events = conn.execute(query, params).fetchall()
    results_count = len(events)
    print(f"[DEBUG] Dashboard fetched {results_count} events with query: {query} and params: {params}")
    if results_count > 0:
        print(f"[DEBUG] First event object: {events[0]}")
        print(f"[DEBUG] First event keys: {list(events[0].keys())}")
    from collections import Counter
    # Compute status counts for chart
    status_counts = Counter(event['status'] for event in events)
    # Ensure all expected keys exist
    for key in ['Upcoming', 'In Progress', 'Completed', 'Cancelled']:
        status_counts.setdefault(key, 0)
    conn.close()
    # Fetch unread notifications for the current user
    conn = get_db_connection()
    notifications = conn.execute('SELECT id, message, is_read FROM notifications WHERE user_id = ? AND is_read = 0', (current_user.id,)).fetchall()
    # Mark notifications as read
    conn.execute('UPDATE notifications SET is_read = 1 WHERE user_id = ?', (current_user.id,))
    conn.commit()
    conn.close()
    return render_template('dashboard.html', stats=stats, events=events, request=request, results_count=results_count, status_counts=status_counts, notifications=notifications, profile=profile, user=current_user)

@app.route('/register', methods=['GET', 'POST'])
def register():
    import re
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        if not username or not email or not password:
            flash('Please fill all fields!')
            return render_template('register.html')
        # Password strength check
        if (len(password) < 8 or
            not re.search(r'[A-Z]', password) or
            not re.search(r'[a-z]', password) or
            not re.search(r'[0-9]', password)):
            flash('Password must be at least 8 characters long and include uppercase, lowercase, and a number!')
            return render_template('register.html')
        if User.query.filter_by(email=email).first():
            flash('Email already registered!')
            return render_template('register.html')
        password_hash = generate_password_hash(password)
        new_user = User(username=username, email=email, password_hash=password_hash, is_admin=False)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/admin_register', methods=['GET', 'POST'])
def admin_register():
    import re
    # Only allow if no admin exists
    admin_exists = User.query.filter_by(is_admin=True).first()
    if admin_exists:
        flash('Admin registration is disabled: an admin already exists.')
        return redirect(url_for('login'))
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        if not username or not email or not password:
            flash('Please fill all fields!')
            return render_template('admin_register.html')
        # Password strength check
        if (len(password) < 8 or
            not re.search(r'[A-Z]', password) or
            not re.search(r'[a-z]', password) or
            not re.search(r'[0-9]', password)):
            flash('Password must be at least 8 characters long and include uppercase, lowercase, and a number!')
            return render_template('admin_register.html')
        if User.query.filter_by(email=email).first():
            flash('Email already registered!')
            return render_template('admin_register.html')
        password_hash = generate_password_hash(password)
        new_admin = User(username=username, email=email, password_hash=password_hash, is_admin=True)
        db.session.add(new_admin)
        db.session.commit()
        flash('Admin account created! Please log in.')
        return redirect(url_for('login'))
    return render_template('admin_register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    try:
        form = LoginForm()
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
        # Use SQLAlchemy to check if any admin exists
        admin_exists = User.query.filter_by(is_admin=True).first() is not None
        # Fetch attendees for face login dropdown
        conn = get_db_connection()
        attendees = conn.execute('SELECT id, name FROM attendees WHERE face_landmarks IS NOT NULL AND face_landmarks != ""').fetchall()
        conn.close()
        if form.validate_on_submit():
            email = form.email.data
            password = form.password.data
            user = User.query.filter_by(email=email).first()
            if user and check_password_hash(user.password_hash, password):
                login_user(user)
                # Log login activity
                from flask import request
                activity = LoginActivity(user_id=user.id, ip_address=request.remote_addr)
                db.session.add(activity)
                db.session.commit()
                flash('Logged in successfully!')
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid email or password!')
        return render_template('login.html', admin_exists=admin_exists, form=form, attendees=attendees)
    except Exception as e:
        import traceback
        logging.error(traceback.format_exc())
        flash('An error occurred during login.')
        return render_template('login.html', admin_exists=True, form=form)

@app.route('/attendee_login', methods=['GET', 'POST'])
def attendee_login():
    from flask import session, render_template, request, redirect, url_for, flash
    try:
        if request.method == 'POST':
            attendee_id = request.form.get('attendee_id')
            if not attendee_id:
                flash('No attendee selected.')
                return redirect(url_for('login'))
            conn = get_db_connection()
            attendee = conn.execute('SELECT * FROM attendees WHERE id = ?', (attendee_id,)).fetchone()
            event = None
            if attendee:
                event = conn.execute('SELECT * FROM events WHERE id = ?', (attendee['event_id'],)).fetchone()
            conn.close()
            if attendee and event:
                session['attendee_id'] = attendee['id']
                photo_uploaded = bool(attendee['face_landmarks'])
                show_welcome = session.get('attendee_verified', False)
                # Reset after use
                session['attendee_verified'] = False
                return render_template('attendee_dashboard.html', attendee=attendee, event=event, photo_uploaded=photo_uploaded, show_welcome=show_welcome)
            else:
                flash('Attendee not found or event missing.')
                return redirect(url_for('login'))
        else:  # GET
            attendee_id = session.get('attendee_id')
            if not attendee_id:
                flash('Please log in as an attendee.')
                return redirect(url_for('login'))
            conn = get_db_connection()
            attendee = conn.execute('SELECT * FROM attendees WHERE id = ?', (attendee_id,)).fetchone()
            event = None
            if attendee:
                event = conn.execute('SELECT * FROM events WHERE id = ?', (attendee['event_id'],)).fetchone()
            conn.close()
            if attendee and event:
                photo_uploaded = bool(attendee['face_landmarks'])
                show_welcome = session.get('attendee_verified', False)
                # Reset after use
                session['attendee_verified'] = False
                return render_template('attendee_dashboard.html', attendee=attendee, event=event, photo_uploaded=photo_uploaded, show_welcome=show_welcome)
            else:
                flash('Attendee not found or event missing.')
                return redirect(url_for('login'))
    except Exception as e:
        import traceback
        logging.error(traceback.format_exc())
        flash('An error occurred during attendee login.')
        return redirect(url_for('login'))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        conn = get_db_connection()
        user = conn.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        conn.close()
        if user:
            flash('A password reset link has been sent to your email address. (Demo: No real email sent)')
        else:
            flash('No account found with that email address.')
    return render_template('forgot_password.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully!')
    return redirect(url_for('login'))

@app.route('/add', methods=['GET', 'POST'])
@login_required
def add_event():
    import datetime
    if request.method == 'POST':
        # Gather form data
        data = {
            'title': request.form['title'],
            'date': request.form['date'],
            'time': request.form['time'],
            'location': request.form['location'],
            'status': request.form['status'],
            'description': request.form['description'],
            'attendance': request.form.get('attendance', 0)
        }
        conn = get_db_connection()
        errors = validate_event_form(data, mode="add", db_conn=conn)
        if errors:
            for error in errors:
                flash(error, 'danger')
            conn.close()
            return render_template('add_event.html')
        # Insert event if all validations pass
        conn.execute('INSERT INTO events (title, date, time, location, status, description, attendance, created_by) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                     (data['title'].strip(), data['date'].strip(), data['time'].strip(), data['location'].strip(), 'Pending Approval', data['description'].strip(), data['attendance'], current_user.id))
        conn.commit()
        conn.close()
        flash('Event added successfully! Notification: New event created.')
        return redirect(url_for('dashboard'))
    return render_template('add_event.html')


@app.route('/edit/<int:event_id>', methods=['GET', 'POST'])
@login_required
def edit_event(event_id):
    import sqlite3
    conn = get_db_connection()
    event = conn.execute('SELECT * FROM events WHERE id = ?', (event_id,)).fetchone()
    if not event:
        conn.close()
        flash('Event not found.')
        return redirect(url_for('dashboard'))
    # Restrict editing to event creator only
    if event['created_by'] != current_user.id:
        conn.close()
        flash('You are not authorized to edit this event.', 'danger')
        return redirect(url_for('dashboard'))
    # Block editing if event is completed
    if event['status'] == 'Completed':
        conn.close()
        flash('Editing is not allowed for completed events.', 'danger')
        return redirect(url_for('dashboard'))
    editing_blocked = event['status'] == 'Completed'
    if request.method == 'POST' and not ('approve_event' in request.form):
        if editing_blocked:
            flash('Editing is not allowed for completed events.')
            conn.close()
            return redirect(url_for('edit_event', event_id=event_id))
    # Only admin can approve events
    if request.method == 'POST' and 'approve_event' in request.form:
        if not hasattr(current_user, 'is_admin') or not current_user.is_admin:
            flash('Only admins can approve events!')
            conn.close()
            return redirect(url_for('edit_event', event_id=event_id))
        conn.execute('UPDATE events SET status = ? WHERE id = ?', ('Upcoming', event_id))
        conn.commit()
        conn.close()
        flash('Event approved and set to Upcoming!')
        return redirect(url_for('edit_event', event_id=event_id))
        flash('Event approved!')
        return redirect(url_for('edit_event', event_id=event_id))
    from datetime import datetime, timedelta
    conn = get_db_connection()
    event = conn.execute('SELECT * FROM events WHERE id = ?', (event_id,)).fetchone()
    attendees = conn.execute('SELECT * FROM attendees WHERE event_id = ?', (event_id,)).fetchall()
    creator_email = None
    if event and 'created_by' in event.keys():
        creator = conn.execute('SELECT email FROM users WHERE id = ?', (event['created_by'],)).fetchone()
        if creator:
            creator_email = creator['email']
    prediction = None
    now = datetime.now()
    event_date = datetime.strptime(event['date'] + ' ' + event['time'], '%Y-%m-%d %H:%M')
    # Block editing for completed or in-progress (after start) events
    if event['status'] == 'Completed':
        conn.close()
        flash('Cannot edit a completed event.')
        return redirect(url_for('dashboard'))
    if event['status'] == 'In Progress' and now >= event_date:
        conn.close()
        flash('Cannot edit an event that is in progress and already started.')
        return redirect(url_for('dashboard'))
    if request.method == 'POST':
        data = {
            'title': request.form['title'],
            'date': request.form['date'],
            'time': request.form['time'],
            'location': request.form['location'],
            'status': request.form['status'],
            'description': request.form['description'],
            'attendance': request.form.get('attendance', 0)
        }
        conn = get_db_connection()
        errors = validate_event_form(data, mode="edit", existing_event_id=event_id, db_conn=conn)
        from datetime import datetime
        now = datetime.now()
        try:
            new_event_dt = datetime.strptime(data['date'] + ' ' + data['time'], '%Y-%m-%d %H:%M')
        except Exception:
            errors.append('Invalid date or time format.')
        if data['status'] == 'Upcoming' and 'Invalid date or time format.' not in errors and new_event_dt <= now:
            errors.append('For upcoming events, date and time must be in the future.')
        if data['status'] == 'In Progress' and 'Invalid date or time format.' not in errors and new_event_dt <= now:
            errors.append('Cannot set event to In Progress if start time has passed.')
        if errors:
            for error in errors:
                flash(error, 'danger')
            conn.close()
            return redirect(request.url)
        conn.execute('UPDATE events SET title=?, date=?, time=?, location=?, status=?, description=?, attendance=? WHERE id=?',
                     (data['title'].strip(), data['date'].strip(), data['time'].strip(), data['location'].strip(), data['status'].strip(), data['description'].strip(), data['attendance'], event_id))
        conn.commit()
        conn.close()
        flash(f"Event updated! Notification: Status is now '{data['status']}'.")
        return redirect(url_for('dashboard'))

    # ML prediction logic
    model, feature_columns = ml_utils.train_attendance_model()
    if model is not None and feature_columns is not None:
        event_features = {
            'date': event['date'],
            'location': event['location'],
            'status': event['status']
        }
        prediction = ml_utils.predict_attendance(event_features, model, feature_columns)
    conn.close()
    from datetime import datetime
    now = datetime.now()
    return render_template('edit_event.html', event=event, attendees=attendees, prediction=prediction, now=now, creator_email=creator_email)

@app.route('/edit/<int:event_id>/add_attendee', methods=['POST'])
@login_required
def add_attendee(event_id):
    import mediapipe as mp
    import cv2
    import numpy as np
    import tempfile
    import base64
    import json
    name = request.form['name']
    email = request.form['email']
    role = request.form['role']
    file = request.files.get('photo')
    img = None
    webcam_photo = request.form.get('webcam_photo')
    # Enforce: must provide either file upload or webcam photo
    if (not file or file.filename == '') and not (webcam_photo and webcam_photo.startswith('data:image')):
        flash('A reference photo is required to add an attendee. Please upload or capture a photo.', 'danger')
        return redirect(request.url)
    if not file or file.filename == '':
        if webcam_photo and webcam_photo.startswith('data:image'):
            header, encoded = webcam_photo.split(',', 1)
            img_bytes = base64.b64decode(encoded)
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp:
            file.save(temp.name)
            img = cv2.imread(temp.name)
    face_landmarks = None
    if img is not None:
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            import logging
            if results.multi_face_landmarks:
                face_landmarks = json.dumps([[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark])
                logging.info(f"[add_attendee] Saved {len(results.multi_face_landmarks[0].landmark)} face landmarks for attendee registration.")
            else:
                flash('No face detected in the provided photo. Please upload a clear face image or try again.', 'danger')
                return redirect(request.url)
    else:
        flash('No image data found. Please upload or capture a photo.', 'danger')
        return redirect(request.url)
    conn = get_db_connection()
    conn.execute("INSERT INTO attendees (event_id, name, email, status, role, previous_attendance_rate, face_landmarks) VALUES (?, ?, ?, ?, ?, ?, ?)",
                 (event_id, name, email, 'Registered', role, 0.0, face_landmarks))
    conn.commit()
    conn.close()
    flash('Attendee added!')
    return redirect(url_for('edit_event', event_id=event_id))

@app.route('/edit/<int:event_id>/edit_attendee/<int:attendee_id>', methods=['POST'])
@login_required
def edit_attendee(event_id, attendee_id):
    name = request.form['edit_name']
    email = request.form['edit_email']
    role = request.form['edit_role']
    conn = get_db_connection()
    conn.execute("UPDATE attendees SET name=?, email=?, role=? WHERE id=?", (name, email, role, attendee_id))
    conn.commit()
    conn.close()
    flash('Attendee updated!')
    return redirect(url_for('edit_event', event_id=event_id))

@app.route('/edit/<int:event_id>/delete_attendee/<int:attendee_id>', methods=['POST'])
@login_required
def delete_attendee(event_id, attendee_id):
    conn = get_db_connection()
    conn.execute("DELETE FROM attendees WHERE id=?", (attendee_id,))
    conn.commit()
    conn.close()
    flash('Attendee deleted!')
    return redirect(url_for('edit_event', event_id=event_id))


@app.route('/face_checkin/<int:event_id>', methods=['GET', 'POST'])
@login_required
def face_checkin(event_id):
    from flask import session, request
    # Support attendee change via query param
    if request.method == 'GET' and request.args.get('change_attendee') == '1':
        session.pop('face_checkin_attendee_id', None)
    """
    Step 1: Attendee selects their name from the list (POST or session)
    Step 2: Only allow face check-in for the selected attendee
    """
    import mediapipe as mp
    import cv2
    import numpy as np
    import tempfile
    import base64
    import json
    from scipy.spatial import distance
    from flask import session
    conn = get_db_connection()
    event = conn.execute('SELECT * FROM events WHERE id = ?', (event_id,)).fetchone()
    if not event:
        conn.close()
        flash('Event not found.', 'danger')
        return redirect(url_for('dashboard'))
    attendees = conn.execute('SELECT * FROM attendees WHERE event_id = ?', (event_id,)).fetchall()
    conn.close()
    attendee_id = session.get('face_checkin_attendee_id')
    if request.method == 'POST':
        # Step 1: If attendee_id not in session or POST, show selection form
        if 'attendee_id' in request.form:
            attendee_id = int(request.form['attendee_id'])
            session['face_checkin_attendee_id'] = attendee_id
        elif not attendee_id:
            return render_template('face_select_attendee.html', event=event, attendees=attendees)
        # Step 2: Handle face check-in for selected attendee
        if 'photo' in request.files or request.form.get('webcam_photo'):
            file = request.files.get('photo')
            img = None
            if not file or file.filename == '':
                webcam_photo = request.form.get('webcam_photo')
                if webcam_photo and webcam_photo.startswith('data:image'):
                    header, encoded = webcam_photo.split(',', 1)
                    img_bytes = base64.b64decode(encoded)
                    img_array = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            else:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp:
                    file.save(temp.name)
                    img = cv2.imread(temp.name)
            if img is not None and attendee_id:
                mp_face_mesh = mp.solutions.face_mesh
                with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
                    results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    import logging
                    if results.multi_face_landmarks:
                        input_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark])
                        logging.info(f"[face_checkin] Extracted {len(input_landmarks)} face landmarks from submitted image.")
                        conn = get_db_connection()
                        attendee = conn.execute('SELECT * FROM attendees WHERE id = ?', (attendee_id,)).fetchone()
                        db_landmarks = attendee['face_landmarks'] if attendee else None
                        match_info = ''
                        if db_landmarks:
                            try:
                                db_landmarks_np = np.array(json.loads(db_landmarks))
                            except Exception:
                                match_info = 'Reference photo is invalid. Please contact admin.'
                                conn.close()
                                return render_template('face_checkin.html', event=event, attendees=attendees, match_info=match_info)
                            if db_landmarks_np.shape == input_landmarks.shape:
                                dist = np.mean(np.linalg.norm(db_landmarks_np - input_landmarks, axis=1))
                                logging.info(f"[face_checkin] Face check-in similarity distance: {dist}")
                                threshold = 0.1
                                if dist < threshold:
                                    conn.execute('UPDATE attendees SET status = ? WHERE id = ?', ('Checked In', attendee_id))
                                    conn.commit()
                                    match_info = f"Face matched! {attendee['name']} has been checked in. (Similarity score: {1-dist/threshold:.2f}, Distance: {dist:.4f})"
                                else:
                                    match_info = 'Face did not match the registered reference. Please try again.'
                                    logging.info(f"[face_checkin] Face check-in failed. Distance: {dist} >= threshold: {threshold}")
                            else:
                                match_info = 'Reference photo is invalid. Please contact admin.'
                        else:
                            match_info = 'No reference photo found for this attendee. Please register your reference photo.'
                        conn.close()
                        return render_template('face_checkin.html', event=event, attendees=attendees, match_info=match_info)
                    else:
                        flash('No face detected in the submitted photo. Please try again.')
                        return redirect(request.url)
        # If not a photo POST, show check-in form
        return render_template('face_checkin.html', event=event, attendees=attendees)
    # GET: If attendee_id not set, show selection form
    if not attendee_id:
        return render_template('face_select_attendee.html', event=event, attendees=attendees)
    return render_template('face_checkin.html', event=event, attendees=attendees)

@app.route('/predict_attendance/<int:event_id>')
@login_required
def predict_attendance_page(event_id):

    conn = get_db_connection()
    event = conn.execute('SELECT * FROM events WHERE id = ?', (event_id,)).fetchone()
    conn.close()
    model, feature_columns = ml_utils.train_attendance_model()
    prediction = None
    if model is not None and feature_columns is not None:
        event_features = {
            'date': event['date'],
            'location': event['location'],
            'status': event['status']
        }
        prediction = ml_utils.predict_attendance(event_features, model, feature_columns)
    return render_template('predict_attendance.html', event=event, prediction=prediction)

@app.route('/delete/<int:event_id>', methods=['POST'])
@login_required
def delete_event(event_id):
    conn = get_db_connection()
    event = conn.execute('SELECT status FROM events WHERE id = ?', (event_id,)).fetchone()
    if event and event['status'] == 'Completed':
        conn.close()
        flash('Cannot delete a completed event.', 'danger')
        return redirect(url_for('dashboard'))
    # Soft delete: set deleted=1
    conn.execute('UPDATE events SET deleted = 1 WHERE id = ?', (event_id,))
    conn.commit()
    conn.close()
    flash('Event deleted. <a href="/undo_delete/{}" class="alert-link">Undo</a>'.format(event_id), 'warning')
    return redirect(url_for('dashboard'))

@app.route('/undo_delete/<int:event_id>')
@login_required
def undo_delete_event(event_id):
    conn = get_db_connection()
    conn.execute('UPDATE events SET deleted = 0 WHERE id = ?', (event_id,))
    conn.commit()
    conn.close()
    flash('Event restored successfully!', 'success')
    return redirect(url_for('dashboard'))
    conn = get_db_connection()
    event = conn.execute('SELECT status FROM events WHERE id = ?', (event_id,)).fetchone()
    if event and event['status'] == 'Completed':
        conn.close()
        flash('Cannot cancel a completed event.')
        return redirect(url_for('dashboard'))
    conn.execute("UPDATE events SET status = 'Cancelled' WHERE id = ?", (event_id,))
    conn.commit()
    conn.close()
    flash('Event cancelled! Notification: Event status set to Cancelled.')
    return redirect(url_for('dashboard'))

# --- Attendee Management ---
@app.route('/event/<int:event_id>/attendees')
@login_required
def view_attendees(event_id):
    # If admin, show face images
    show_faces = hasattr(current_user, 'is_admin') and current_user.is_admin
    conn = get_db_connection()
    event = conn.execute('SELECT * FROM events WHERE id = ?', (event_id,)).fetchone()
    attendees = conn.execute('SELECT * FROM attendees WHERE event_id = ?', (event_id,)).fetchall()
    conn.close()
    return render_template('attendees.html', event=event, attendees=attendees, show_faces=show_faces)

@app.route('/admin/attendees')
@login_required
def admin_view_attendees():
    if not hasattr(current_user, 'is_admin') or not current_user.is_admin:
        flash('Access denied: Admins only!')
        return redirect(url_for('dashboard'))
    conn = get_db_connection()
    attendees = conn.execute('SELECT * FROM attendees').fetchall()
    conn.close()
    return render_template('admin_attendees.html', attendees=attendees)

@app.route('/attendees/<int:attendee_id>/update_photo', methods=['GET', 'POST'])
def update_attendee_photo(attendee_id):
    # Allow if user is logged in, or if attendee_id matches session['attendee_id'] (set after attendee login)
    from flask import session, redirect, url_for, flash
    if not session.get('user_id'):
        # Not logged in as user, must be logged in as attendee
        if session.get('attendee_id') != attendee_id:
            flash('Access denied. Please log in as this attendee to update the photo.')
            return redirect(url_for('login'))
    import mediapipe as mp
    import cv2
    import numpy as np
    import tempfile
    import base64
    import json
    conn = get_db_connection()
    attendee = conn.execute('SELECT * FROM attendees WHERE id = ?', (attendee_id,)).fetchone()
    if request.method == 'POST':
        file = request.files.get('photo')
        face_landmarks = None
        img = None
        if not file or file.filename == '':
            webcam_photo = request.form.get('webcam_photo')
            if webcam_photo:
                header, encoded = webcam_photo.split(',', 1)
                img_bytes = base64.b64decode(encoded)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        else:
            temp = tempfile.NamedTemporaryFile(delete=False)
            file.save(temp.name)
            img = cv2.imread(temp.name)
            temp.close()
        if img is not None:
            mp_face_mesh = mp.solutions.face_mesh
            with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
                results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                import logging
                if results.multi_face_landmarks:
                    new_landmarks = [[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark]
                    # Compare to reference face_landmarks from registration
                    if attendee['face_landmarks']:
                        import numpy as np
                        ref_landmarks = np.array(json.loads(attendee['face_landmarks']))
                        new_landmarks_arr = np.array(new_landmarks)
                        # Use only first 468 points for both (mediapipe default)
                        ref_landmarks = ref_landmarks[:468]
                        new_landmarks_arr = new_landmarks_arr[:468]
                        # Compute Euclidean distance
                        dist = np.linalg.norm(ref_landmarks - new_landmarks_arr)
                        import logging
                        logging.info(f"[attendee_photo_verification] Distance: {dist}")
                        # Lower threshold for stricter verification
                        if dist < 0.1:  # Strict threshold, tune as needed
                            from flask import session
                            session['attendee_verified'] = True
                            flash('Welcome to the event! Your face has been recognized.', 'success')
                            event_id = attendee['event_id']
                            conn.close()
                            return redirect(url_for('attendee_login'))
                        else:
                            from flask import session
                            session['attendee_verified'] = False
                            flash('Face does not match our records. Please try again or contact the event creator for more details.', 'danger')
                            conn.close()
                            return redirect(url_for('login'))
                    else:
                        # No reference yet, allow first upload and save
                        face_landmarks = json.dumps(new_landmarks)
                        conn.execute('UPDATE attendees SET face_landmarks = ? WHERE id = ?', (face_landmarks, attendee_id))
                        conn.commit()
                        flash('Reference photo updated successfully!')
                        event_id = attendee['event_id']
                        conn.close()
                        return redirect(url_for('attendee_login'))
                else:
                    flash('No face detected in the provided photo. Please try again.')
                    return redirect(request.url)
    conn.close()
    return render_template('update_attendee_photo.html', attendee=attendee)

@app.route('/calendar')
def calendar_view():
    return render_template('calendar.html')

@app.route('/api/upcoming_events')
def api_upcoming_events():
    conn = get_db_connection()
    # Assuming your 'events' table has a 'date' column and stores future events
    events = conn.execute(
        "SELECT id, title, date FROM events WHERE date >= DATE('now') ORDER BY date ASC"
    ).fetchall()
    conn.close()
    event_list = []
    for event in events:
        event_list.append({
            'id': event['id'],
            'title': event['title'],
            'date': event['date']
        })
    return jsonify({'events': event_list})

@app.route('/api/events')
def api_events():
    conn = get_db_connection()
    events = conn.execute('SELECT * FROM events').fetchall()
    conn.close()
    # Convert events to FullCalendar format
    event_list = []
    for event in events:
        event_list.append({
            'id': event['id'],
            'title': event['title'],
            'start': event['date'],
            'url': url_for('edit_event', event_id=event['id'])
        })
    return jsonify(event_list)

@app.route('/admin')
@login_required
def admin_dashboard():
    if not hasattr(current_user, 'is_admin') or not current_user.is_admin:
        flash('Access denied: Admins only!')
        return redirect(url_for('dashboard'))
    import datetime
    today = datetime.date.today().isoformat()
    conn = get_db_connection()
    # Auto-complete events whose date has passed
    conn.execute("UPDATE events SET status = 'Completed' WHERE (status = 'Upcoming' OR status = 'In Progress') AND date <= ?", (today,))
    conn.commit()
    users = conn.execute('SELECT id, email, is_admin FROM users').fetchall()
    events = conn.execute('SELECT * FROM events').fetchall()
    # Calculate dashboard stats
    stats = {
        'total_events': conn.execute('SELECT COUNT(*) FROM events').fetchone()[0],
        'upcoming_events': conn.execute("SELECT COUNT(*) FROM events WHERE status = 'Upcoming'").fetchone()[0],
        'completed_events': conn.execute("SELECT COUNT(*) FROM events WHERE status = 'Completed'").fetchone()[0],
        'cancelled_events': conn.execute("SELECT COUNT(*) FROM events WHERE status = 'Cancelled'").fetchone()[0],
        'pending_approval_events': conn.execute("SELECT COUNT(*) FROM events WHERE status = 'Pending Approval'").fetchone()[0],
        'total_users': conn.execute('SELECT COUNT(*) FROM users').fetchone()[0],
    }
    conn.close()
    model_accuracy = session.get('model_accuracy')
    model_metrics = session.get('model_metrics')
    cm_img = session.get('cm_img')
    roc_img = session.get('roc_img')
    # Load ML model and feature importances
    model, feature_columns = ml_utils.train_attendance_model()
    feature_importances = []
    if model is not None and feature_columns is not None:
        feature_importances = ml_utils.get_feature_importances(model, feature_columns)
    return render_template('admin_dashboard.html', users=users, events=events, stats=stats, model_accuracy=model_accuracy, model_metrics=model_metrics, cm_img=cm_img, roc_img=roc_img, feature_importances=feature_importances)

@app.route('/admin/user_action', methods=['POST'])
@login_required
def admin_user_action():
    if not hasattr(current_user, 'is_admin') or not current_user.is_admin:
        flash('Access denied: Admins only!')
        return redirect(url_for('dashboard'))
    user_id = request.form.get('user_id')
    action = request.form.get('action')
    if not user_id or not action:
        flash('Invalid request.')
        return redirect(url_for('admin_dashboard'))
    conn = get_db_connection()
    if action == 'promote':
        conn.execute('UPDATE users SET is_admin = 1 WHERE id = ?', (user_id,))
        flash('User promoted to admin.')
    elif action == 'demote':
        conn.execute('UPDATE users SET is_admin = 0 WHERE id = ?', (user_id,))
        flash('User demoted from admin.')
    elif action == 'delete':
        conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
        flash('User deleted.')
    conn.commit()
    conn.close()
    return redirect(url_for('admin_dashboard'))

# --- Admin: Clear All Images ---
@app.route('/admin/clear_images', methods=['POST'])
@login_required
def admin_clear_images():
    if not hasattr(current_user, 'is_admin') or not current_user.is_admin:
        flash('Access denied: Admins only!')
        return redirect(url_for('dashboard'))
    try:
        conn = get_db_connection()
        conn.execute('UPDATE attendees SET face_landmarks = NULL')
        conn.commit()
        conn.close()
        flash('All attendee face images have been cleared from the database.', 'success')
    except Exception as e:
        flash(f'Error clearing images: {e}', 'danger')
    return redirect(url_for('admin_dashboard'))

# --- User Profile ---
@app.route('/profile')
@login_required
def profile():
    # Fetch recent login activity
    activity = LoginActivity.query.filter_by(user_id=current_user.id).order_by(LoginActivity.timestamp.desc()).limit(5).all()
    profile = Profile.query.filter_by(user_id=current_user.id).first()
    return render_template('profile.html', user=current_user, activity=activity, profile=profile)

# --- Profile: Update Profile Picture ---
@app.route('/profile/update_picture', methods=['POST'])
@login_required
def update_profile_picture():
    if 'profile_picture' not in request.files:
        flash('No file part.', 'danger')
        return redirect(url_for('profile'))
    file = request.files['profile_picture']
    if file.filename == '':
        flash('No selected file.', 'danger')
        return redirect(url_for('profile'))
    if file and (file.filename.lower().endswith('.png') or file.filename.lower().endswith('.jpg') or file.filename.lower().endswith('.jpeg') or file.filename.lower().endswith('.gif')):
        import secrets
        from werkzeug.utils import secure_filename
        import os
        filename = secure_filename(file.filename)
        unique_hex = secrets.token_hex(8)
        _, ext = os.path.splitext(filename)
        pic_filename = f"{current_user.id}_{unique_hex}{ext}"
        pic_path = os.path.join(app.root_path, 'static', 'profile_pics', pic_filename)
        os.makedirs(os.path.dirname(pic_path), exist_ok=True)
        file.save(pic_path)
        # Save to profile
        profile = Profile.query.filter_by(user_id=current_user.id).first()
        if not profile:
            profile = Profile(user_id=current_user.id)
            db.session.add(profile)
        profile.profile_picture = pic_filename
        db.session.commit()
        flash('Profile picture updated!', 'success')
    else:
        flash('Invalid file type.', 'danger')
    return redirect(url_for('profile'))

# --- Profile: Update Profile Info ---
@app.route('/profile/update_info', methods=['POST'])
@login_required
def update_profile_info():
    name = request.form.get('name')
    phone = request.form.get('phone')
    address = request.form.get('address')
    gender = request.form.get('gender')
    city = request.form.get('city')
    profile = Profile.query.filter_by(user_id=current_user.id).first()
    if not profile:
        profile = Profile(user_id=current_user.id)
        db.session.add(profile)
    profile.name = name
    profile.phone = phone
    profile.address = address
    profile.gender = gender
    profile.city = city
    db.session.commit()
    flash('Profile information updated!', 'success')
    return redirect(url_for('profile'))

# --- Profile: Update Display Name ---
@app.route('/profile/update_name', methods=['POST'])
@login_required
def update_display_name():
    new_name = request.form.get('display_name', '').strip()
    if not new_name:
        flash('Display name cannot be empty.', 'danger')
        return redirect(url_for('profile'))
    if User.query.filter_by(username=new_name).first() and new_name != current_user.username:
        flash('Display name already taken.', 'danger')
        return redirect(url_for('profile'))
    current_user.username = new_name
    db.session.commit()
    flash('Display name updated.', 'success')
    return redirect(url_for('profile'))

# --- Profile: Update Email ---
@app.route('/profile/update_email', methods=['POST'])
@login_required
def update_email():
    new_email = request.form.get('email', '').strip()
    if not new_email:
        flash('Email cannot be empty.', 'danger')
        return redirect(url_for('profile'))
    if User.query.filter_by(email=new_email).first() and new_email != current_user.email:
        flash('Email already in use.', 'danger')
        return redirect(url_for('profile'))
    current_user.email = new_email
    db.session.commit()
    flash('Email updated.', 'success')
    return redirect(url_for('profile'))

# --- Profile: Change Password ---
@app.route('/profile/change_password', methods=['POST'])
@login_required
def change_password():
    from werkzeug.security import generate_password_hash, check_password_hash
    new_pw = request.form.get('new_password', '')
    confirm_pw = request.form.get('confirm_password', '')
    if not new_pw or not confirm_pw:
        flash('Please fill out both password fields.', 'danger')
        return redirect(url_for('profile'))
    if new_pw != confirm_pw:
        flash('Passwords do not match.', 'danger')
        return redirect(url_for('profile'))
    if len(new_pw) < 6:
        flash('Password must be at least 6 characters.', 'danger')
        return redirect(url_for('profile'))
    current_user.password_hash = generate_password_hash(new_pw)
    db.session.commit()
    flash('Password changed successfully.', 'success')
    return redirect(url_for('profile'))

# --- Profile: Toggle Notifications ---
@app.route('/profile/toggle_notifications', methods=['POST'])
@login_required
def toggle_notifications():
    current_user.notification_enabled = not current_user.notification_enabled
    db.session.commit()
    flash('Notification preferences updated.', 'success')
    return redirect(url_for('profile'))

# --- Profile: Toggle Theme ---

# --- User: Cancel Event ---
@app.route('/cancel/<int:event_id>', methods=['POST'])
@login_required
def cancel_event(event_id):
    event = db.session.execute(db.select(Event).where(Event.id == event_id)).scalar_one_or_none()
    if not event:
        flash('Event not found.', 'danger')
        return redirect(url_for('dashboard'))
    # Only admins can cancel events since creator is not tracked
    if not current_user.is_admin:
        flash('You do not have permission to cancel this event.', 'danger')
        return redirect(url_for('dashboard'))
    if event.status == 'Cancelled':
        flash('Event is already cancelled.', 'info')
        return redirect(url_for('dashboard'))
    event.status = 'Cancelled'
    db.session.commit()
    # Update the SQLite DB directly for dashboard
    conn = get_db_connection()
    conn.execute('UPDATE events SET status = ? WHERE id = ?', ('Cancelled', event_id))
    conn.commit()
    conn.close()
    flash('Event cancelled successfully.', 'success')
    return redirect(url_for('dashboard'))
@app.route('/profile/toggle_theme', methods=['POST'])
@login_required
def toggle_theme():
    current_user.theme = 'dark' if current_user.theme == 'light' else 'light'
    db.session.commit()
    flash(f"Theme changed to {current_user.theme} mode.", 'success')
    return redirect(url_for('profile'))

# --- Admin: Profile View & Update ---
@app.route('/admin/profile/upload_picture', methods=['POST'])
@login_required
def admin_upload_profile_picture():
    if not current_user.is_admin:
        flash('Access denied. Admins only.', 'danger')
        return redirect(url_for('dashboard'))
    if 'profile_picture' not in request.files:
        flash('No file part.', 'danger')
        return redirect(url_for('admin_profile'))
    file = request.files['profile_picture']
    if file.filename == '':
        flash('No selected file.', 'danger')
        return redirect(url_for('admin_profile'))
    if file and (file.filename.lower().endswith('.png') or file.filename.lower().endswith('.jpg') or file.filename.lower().endswith('.jpeg') or file.filename.lower().endswith('.gif')):
        import secrets
        pic_filename = f"admin_{current_user.id}_" + secrets.token_hex(8) + '.' + file.filename.rsplit('.', 1)[1].lower()
        pic_path = os.path.join(app.root_path, 'static', 'profile_pics', pic_filename)
        os.makedirs(os.path.dirname(pic_path), exist_ok=True)
        file.save(pic_path)
        profile = Profile.query.filter_by(user_id=current_user.id).first()
        if not profile:
            profile = Profile(user_id=current_user.id)
            db.session.add(profile)
        profile.profile_picture = pic_filename
        db.session.commit()
        flash('Profile picture updated!', 'success')
    else:
        flash('Invalid file type.', 'danger')
    return redirect(url_for('admin_profile'))

@app.route('/admin/profile', methods=['GET', 'POST'])
@login_required
def admin_profile():
    if not current_user.is_admin:
        flash('Access denied. Admins only.', 'danger')
        return redirect(url_for('dashboard'))
    profile = Profile.query.filter_by(user_id=current_user.id).first()
    if request.method == 'POST':
        name = request.form.get('name')
        phone = request.form.get('phone')
        address = request.form.get('address')
        gender = request.form.get('gender')
        city = request.form.get('city')
        if not profile:
            profile = Profile(user_id=current_user.id)
            db.session.add(profile)
        profile.name = name
        profile.phone = phone
        profile.address = address
        profile.gender = gender
        profile.city = city
        db.session.commit()
        flash('Admin profile updated!', 'success')
        return redirect(url_for('admin_profile'))
    return render_template('admin_profile.html', profile=profile)

# --- Admin: View Any User Profile ---
@app.route('/admin/user/<int:user_id>/profile')
@login_required
def admin_user_profile(user_id):
    if not current_user.is_admin:
        flash('Access denied. Admins only.', 'danger')
        return redirect(url_for('dashboard'))
    user = User.query.get_or_404(user_id)
    profile = Profile.query.filter_by(user_id=user_id).first()
    return render_template('admin_user_profile.html', user=user, profile=profile)

# --- Profile: Account Deletion ---
@app.route('/profile/delete_account', methods=['POST'])
@login_required
def delete_account():
    user_id = current_user.id
    logout_user()
    user = User.query.get(user_id)
    db.session.delete(user)
    db.session.commit()
    flash('Your account has been deleted.', 'success')
    return redirect(url_for('login'))

@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_message = request.json.get('message', '').lower()
    # Simple rule-based AI assistant
    if any(greet in user_message for greet in ['hello', 'hi', 'hey']):
        reply = "Hello! How can I help you with your event management today?"
    elif 'help' in user_message:
        reply = "You can ask me about events, attendance, registration, or anything related to this system. For example, try: 'How do I add an event?' or 'Show me upcoming events.'"
    elif 'event' in user_message and 'add' in user_message:
        reply = "To add an event, click the 'Add Event' button on your dashboard and fill in the details."
    elif 'attendance' in user_message:
        reply = "You can predict attendance or manage attendees from the dashboard. Would you like steps for a specific feature?"
    elif 'register' in user_message:
        reply = "To register, click the 'Register' link in the navigation bar and fill out the form."
    elif 'thank' in user_message:
        reply = "You're welcome! Let me know if you have any more questions."
    elif user_message.strip() == '':
        reply = "Please type a message."
    else:
        reply = "I'm your event assistant bot. You can ask me about events, attendance, or using the system. Type 'help' to see what I can do!"
    return jsonify({"response": reply})

@app.route('/admin/event_action', methods=['POST'])
@login_required
def admin_event_action():
    if not hasattr(current_user, 'is_admin') or not current_user.is_admin:
        flash('Access denied: Admins only!')
        return redirect(url_for('dashboard'))
    event_id = request.form.get('event_id')
    action = request.form.get('action')
    if not event_id or not action:
        flash('Invalid request.')
        return redirect(url_for('admin_dashboard'))
    conn = get_db_connection()
    if action == 'delete':
        conn.execute('DELETE FROM events WHERE id = ?', (event_id,))
        flash('Event deleted.')
    elif action == 'cancel':
        conn.execute("UPDATE events SET status = 'Cancelled' WHERE id = ?", (event_id,))
        flash('Event cancelled! Notification: Event status set to Cancelled.')
    elif action == 'approve':
        conn.execute("UPDATE events SET status = 'Upcoming' WHERE id = ?", (event_id,))
        # Notify event creator
        creator = conn.execute('SELECT created_by FROM events WHERE id = ?', (event_id,)).fetchone()
        if creator:
            user = conn.execute('SELECT id, username, email FROM users WHERE id = ?', (creator['created_by'],)).fetchone()
            if user:
                # Store notification in notifications table
                conn.execute('INSERT INTO notifications (user_id, message, is_read) VALUES (?, ?, 0)', (user['id'], f"Your event has been approved!"))
                # Send email notification
                try:
                    msg = Message('Your Event Has Been Approved', recipients=[user['email']])
                    msg.body = f"Hello {user['username']},\n\nYour event has been approved and is now live!\n\nThank you for using Event Manager."
                    mail.send(msg)
                except Exception as e:
                    flash(f"Failed to send email notification: {e}", 'warning')
                flash(f"Notification: Event approved! User '{user['username']}' ({user['email']}) has been notified.")
        flash('Event approved! Notification: Event status set to Upcoming.')
    else:
        flash('Unknown action.')
    conn.commit()
    conn.close()
    return redirect(url_for('admin_dashboard'))

@app.route('/retrain_model', methods=['POST'])
@login_required
def retrain_model():
    flash('Retrain route called (debug).')
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, roc_curve, auc
    # Use extract_ml_data if available, else fallback
    try:
        X, y = ml_utils.extract_ml_data()
    except AttributeError:
        df = ml_utils.get_event_data()
        # Example fallback: use attendance as y, and the rest as X (customize as needed)
        if 'attendance' in df.columns:
            y = df['attendance']
            X = df.drop(columns=['attendance'])
        else:
            flash('No extract_ml_data function and cannot fallback. Please check ML utilities.')
            return redirect(url_for('admin_dashboard'))
    if X.shape[0] < 10:
        flash('Not enough data to train model. Add more attendee/event records.')
        return redirect(url_for('admin_dashboard'))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    acc, report = ml_utils.train_and_evaluate_model(X_train, y_train, X_test, y_test)
    # Save confusion matrix and ROC curve as base64
    y_pred = ml_utils.load_model().predict(X_test)
    y_score = getattr(ml_utils.load_model(), "predict_proba", lambda x: None)(X_test)
    cm_img = ml_visuals.plot_confusion_matrix(y_test, y_pred, labels=[0,1])
    roc_img = None
    if y_score is not None:
        roc_img = ml_visuals.plot_roc_curve(y_test, y_score[:,1])
    session['model_accuracy'] = f'{acc:.2%}'
    session['model_metrics'] = report
    session['cm_img'] = cm_img
    session['roc_img'] = roc_img
    flash(f'Model retrained on real data. Accuracy: {acc:.2%}')
    return redirect(url_for('admin_dashboard'))

@app.route('/predict_attendance', methods=['POST'])
@login_required
def predict_attendance():
    event_id = request.form.get('event_id')
    if not event_id:
        return jsonify({'error':'Missing event_id'}), 400
    preds = ml_utils.predict_attendance_for_event(event_id)
    # Fetch attendee names
    conn = get_db_connection()
    attendees = conn.execute('SELECT id, name FROM attendees WHERE event_id = ?', (event_id,)).fetchall()
    conn.close()
    id_to_name = {a['id']: a['name'] for a in attendees}
    pred_table = [(id_to_name.get(att_id, att_id), 'Present' if status==1 else 'Absent') for att_id, status in preds]
    return jsonify({'predictions': pred_table})

@app.route('/download_metrics')
@login_required
def download_metrics():
    metrics = session.get('model_metrics')
    if not metrics:
        return 'No metrics available', 400
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Label','Precision','Recall','F1'])
    for label, m in metrics.items():
        if label in ['accuracy','macro avg','weighted avg']:
            continue
        writer.writerow([label, m['precision'], m['recall'], m['f1-score']])
    response = make_response(output.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=model_metrics.csv'
    response.headers['Content-Type'] = 'text/csv'
    return response

@app.route('/download_predictions/<int:event_id>')
@login_required
def download_predictions(event_id):
    if not hasattr(current_user, 'is_admin') or not current_user.is_admin:
        flash('Only admins can download predictions!')
        return redirect(url_for('dashboard'))
    print(f"[DEBUG] download_predictions called for event_id={event_id}")
    try:
        preds = ml_utils.predict_attendance_for_event(event_id)
        print(f"[DEBUG] Predictions: {preds}")
        conn = get_db_connection()
        attendees = conn.execute('SELECT id, name FROM attendees WHERE event_id = ?', (event_id,)).fetchall()
        conn.close()
        id_to_name = {a['id']: a['name'] for a in attendees}
        print(f"[DEBUG] id_to_name mapping: {id_to_name}")
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['Attendee','Predicted Status'])
        for att_id, status in preds:
            print(f"[DEBUG] Writing row: {att_id}, {status}")
            writer.writerow([id_to_name.get(att_id, att_id), 'Present' if status==1 else 'Absent'])
        response = make_response(output.getvalue())
        response.headers['Content-Disposition'] = f'attachment; filename=predictions_event_{event_id}.csv'
        response.headers['Content-Type'] = 'text/csv'
        print("[DEBUG] Returning CSV response")
        return response
    except Exception as e:
        logging.error(f"Exception in download_predictions: {e}")
        return f"Error in download_predictions: {e}", 500

@app.route('/ml_vis/<imgtype>')
@login_required
def ml_vis(imgtype):
    img = session.get(f'{imgtype}_img')
    if not img:
        return 'No image available', 404
    return f'<img src="data:image/png;base64,{img}" />'

# --- Public Home, Guest Login, and Attendee Face Login ---
@app.route('/', methods=['GET'])
def root():
    from flask_login import current_user
    if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/guest_face_checkin', methods=['GET', 'POST'])
def guest_face_checkin():
    import mediapipe as mp
    import cv2
    import numpy as np
    import base64
    import json
    import logging
    if request.method == 'GET':
        return render_template('guest_face_checkin.html')
    image_data = request.form.get('image_data')
    if not image_data:
        flash('Missing face image.')
        return redirect(url_for('guest_face_checkin'))
    # Decode image from base64
    try:
        header, encoded = image_data.split(',', 1)
        img_bytes = base64.b64decode(encoded)
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        logging.error(f"[guest_face_checkin] Failed to decode image: {e}")
        flash('Invalid image data.')
        return redirect(url_for('guest_face_checkin'))
    # Extract face landmarks from image
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            flash('No face detected in the submitted photo. Please try again.')
            return redirect(url_for('guest_face_checkin'))
        input_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark])
        logging.info(f"[guest_face_checkin] Extracted {len(input_landmarks)} face landmarks from submitted image.")
        # Compare to all registered attendees
        conn = get_db_connection()
        attendees = conn.execute('SELECT * FROM attendees WHERE face_landmarks IS NOT NULL AND face_landmarks != ""').fetchall()
        conn.close()
        best_match = None
        best_dist = float('inf')
        for attendee in attendees:
            db_landmarks = attendee['face_landmarks']
            try:
                db_landmarks_np = np.array(json.loads(db_landmarks))
            except Exception as e:
                continue
            if db_landmarks_np.shape == input_landmarks.shape:
                dist = np.mean(np.linalg.norm(db_landmarks_np - input_landmarks, axis=1))
                if dist < best_dist:
                    best_dist = dist
                    best_match = attendee
        threshold = 0.1
        if best_match and best_dist < threshold:
            logging.info(f"[guest_face_checkin] Guest matched attendee {best_match['name']} with distance {best_dist}")
            return render_template('face_login_success.html', attendee=best_match, event=None, guest_mode=True)
        else:
            error_message = 'Face did not match any registered attendee. Please try again.'
            logging.info(f"[guest_face_checkin] No match found. Best distance: {best_dist}")
            return render_template('face_login_error.html', error_message=error_message, guest_mode=True)

@app.route('/guest_login', methods=['POST'])
def guest_login():
    # Start a guest session; could set session['guest'] = True or similar
    session['guest'] = True
    flash('Signed in as guest!')
    return redirect(url_for('dashboard'))

@app.route('/attendee_face_login', methods=['POST'])
def attendee_face_login():
    import mediapipe as mp
    import cv2
    import numpy as np
    import base64
    import json
    import logging

    attendee_id = request.form.get('attendee_id')
    face_image = request.form.get('face_image')
    if not attendee_id:
        flash('Please select your name.')
        return redirect(url_for('home'))
    if face_image:
        # Handle direct face image submission (from modal)
        try:
            header, encoded = face_image.split(',', 1)
            img_bytes = base64.b64decode(encoded)
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            logging.error(f"[face_login] Error decoding image: {e}")
            error_message = 'Invalid image data.'
            return render_template('face_login_error.html', error_message=error_message)
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
            results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if results.multi_face_landmarks:
                input_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark])
                logging.info(f"[face_login] Extracted {len(input_landmarks)} face landmarks from submitted image.")
                # Retrieve attendee's reference landmarks
                conn = get_db_connection()
                attendee = conn.execute('SELECT * FROM attendees WHERE id = ?', (attendee_id,)).fetchone()
                event = None
                if attendee:
                    event = conn.execute('SELECT * FROM events WHERE id = ?', (attendee['event_id'],)).fetchone()
                db_landmarks = attendee['face_landmarks'] if attendee else None
                conn.close()
                if not attendee or not event:
                    error_message = 'Attendee or event not found.'
                    return render_template('face_login_error.html', error_message=error_message)
                if db_landmarks:
                    try:
                        db_landmarks_np = np.array(json.loads(db_landmarks))
                    except Exception:
                        error_message = 'Reference photo is invalid. Please contact admin.'
                        return render_template('face_login_error.html', error_message=error_message)
                    if db_landmarks_np.shape == input_landmarks.shape:
                        dist = np.mean(np.linalg.norm(db_landmarks_np - input_landmarks, axis=1))
                        logging.info(f"[face_login] Face match similarity distance: {dist}")
                        threshold = 0.1  # strict threshold for login
                        if dist < threshold:
                            # Match!
                            return render_template('face_login_success.html', attendee=attendee, event=event)
                        else:
                            # Mismatch!
                            error_message = f'Face did not match the registered reference. (Distance: {dist:.4f}) Please try again.'
                            return render_template('face_login_error.html', error_message=error_message)
                    else:
                        error_message = 'Reference photo is invalid. Please contact admin.'
                        return render_template('face_login_error.html', error_message=error_message)
                else:
                    error_message = 'No reference photo found for this attendee. Please contact admin.'
                    return render_template('face_login_error.html', error_message=error_message)
            else:
                error_message = 'No face detected in the submitted photo. Please try again.'
                return render_template('face_login_error.html', error_message=error_message)
    else:
        # Fallback to legacy session-based flow if no image provided
        session['face_checkin_attendee_id'] = attendee_id
        return redirect(url_for('face_checkin_attendee'))

@app.route('/face_checkin_attendee', methods=['GET', 'POST'])
def face_checkin_attendee():
    attendee_id = session.get('face_checkin_attendee_id')
    if not attendee_id:
        flash('No attendee selected.')
        return redirect(url_for('login'))
    # Fetch attendee info
    conn = get_db_connection()
    attendee = conn.execute('SELECT * FROM attendees WHERE id = ?', (attendee_id,)).fetchone()
    conn.close()
    if not attendee or not attendee['face_landmarks']:
        flash('Selected attendee does not have face data.')
        return redirect(url_for('login'))
    return render_template('face_login.html', attendee=attendee)

@app.route('/submit_face_login', methods=['POST'])
def submit_face_login():
    import mediapipe as mp
    import cv2
    import numpy as np
    import base64
    import json
    import logging
    attendee_id = session.get('face_checkin_attendee_id')
    image_data = request.form.get('image_data')
    if not attendee_id or not image_data:
        flash('Missing face image or attendee information.')
        return redirect(url_for('login'))
    # Decode image from base64
    header, encoded = image_data.split(',', 1)
    img_bytes = base64.b64decode(encoded)
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    # Full face recognition: extract face landmarks and compare
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            input_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.multi_face_landmarks[0].landmark])
            logging.info(f"[face_login] Extracted {len(input_landmarks)} face landmarks from submitted image.")
            # Retrieve attendee's reference landmarks
            conn = get_db_connection()
            attendee = conn.execute('SELECT * FROM attendees WHERE id = ?', (attendee_id,)).fetchone()
            event = None
            if attendee:
                event = conn.execute('SELECT * FROM events WHERE id = ?', (attendee['event_id'],)).fetchone()
            db_landmarks = attendee['face_landmarks'] if attendee else None
            conn.close()
            if not attendee or not event:
                error_message = 'Attendee or event not found.'
                return render_template('face_login_error.html', error_message=error_message)
            if db_landmarks:
                try:
                    db_landmarks_np = np.array(json.loads(db_landmarks))
                except Exception:
                    error_message = 'Reference photo is invalid. Please contact admin.'
                    return render_template('face_login_error.html', error_message=error_message)
                if db_landmarks_np.shape == input_landmarks.shape:
                    dist = np.mean(np.linalg.norm(db_landmarks_np - input_landmarks, axis=1))
                    logging.info(f"[face_login] Face match similarity distance: {dist}")
                    threshold = 0.1  # strict threshold for login
                    if dist < threshold:
                        # Match!
                        return render_template('face_login_success.html', attendee=attendee, event=event)
                    else:
                        # Mismatch!
                        error_message = f'Face did not match the registered reference. (Distance: {dist:.4f}) Please try again.'
                        return render_template('face_login_error.html', error_message=error_message)
                else:
                    error_message = 'Reference photo is invalid. Please contact admin.'
                    return render_template('face_login_error.html', error_message=error_message)
            else:
                error_message = 'No reference photo found for this attendee. Please contact admin.'
                return render_template('face_login_error.html', error_message=error_message)
        else:
            flash('No face detected in the submitted photo. Please try again.')
            return redirect(url_for('face_checkin_attendee'))

if __name__ == '__main__':
    app.run(debug=True)
