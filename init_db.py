import sqlite3

conn = sqlite3.connect('events.db')
c = conn.cursor()

# Create users table
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    is_admin INTEGER DEFAULT 0
)
''')

# Create events table
c.execute('''
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    date TEXT,
    time TEXT,
    location TEXT,
    status TEXT,
    description TEXT,
    attendance INTEGER
)
''')

# Create attendees table
c.execute('''
CREATE TABLE IF NOT EXISTS attendees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id INTEGER,
    name TEXT,
    email TEXT,
    status TEXT,
    face_landmarks TEXT,
    role TEXT,
    previous_attendance_rate REAL,
    FOREIGN KEY(event_id) REFERENCES events(id)
)
''')

conn.commit()
conn.close()
print("Database initialized!")
