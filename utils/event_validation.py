# event_validation.py
# Centralized event validation logic for add/edit event flows
import datetime

def validate_event_form(data, mode="add", existing_event_id=None, db_conn=None):
    """
    Validate event form data for add or edit.
    data: dict with keys: title, date, time, location, status, description, attendance
    mode: 'add' or 'edit'
    existing_event_id: For edit mode, skip duplicate check for this event
    db_conn: sqlite3 connection (required for duplicate check)
    Returns: list of error messages (empty if valid)
    """
    errors = []
    title = data.get('title', '').strip()
    date = data.get('date', '').strip()
    time = data.get('time', '').strip()
    location = data.get('location', '').strip()
    status = data.get('status', '').strip()
    description = data.get('description', '').strip()
    attendance = data.get('attendance', 0)
    # Required fields
    if not title or not date or not time or not location or not status:
        errors.append('All fields except description are required.')
    # Title validation
    if len(title) < 3 or len(title) > 100:
        errors.append('Title must be between 3 and 100 characters.')
    # Date validation
    try:
        event_date = datetime.datetime.strptime(date, '%Y-%m-%d').date()
        today = datetime.date.today()
        if event_date < today:
            errors.append('Event date cannot be in the past.')
    except ValueError:
        errors.append('Invalid date format.')
    # Time validation
    try:
        datetime.datetime.strptime(time, '%H:%M')
    except ValueError:
        errors.append('Invalid time format. Use HH:MM (24-hour).')
    # Attendance validation
    try:
        attendance = int(attendance)
        if attendance < 0:
            errors.append('Attendance must be a positive integer.')
    except ValueError:
        errors.append('Attendance must be a positive integer.')
    # Duplicate check
    if db_conn is not None:
        query = 'SELECT id FROM events WHERE date = ? AND time = ? AND LOWER(location) = ?'
        params = (date, time, location.lower())
        rows = db_conn.execute(query, params).fetchall()
        if mode == 'add' and rows:
            errors.append('An event at this date, time, and location already exists.')
        elif mode == 'edit':
            for row in rows:
                if row['id'] != existing_event_id:
                    errors.append('Another event at this date, time, and location already exists.')
                    break
    # Status logic for edit
    if mode == 'edit':
        # Only allow certain status transitions
        if status == 'Completed':
            errors.append('Cannot set status to Completed from edit page.')
    return errors
