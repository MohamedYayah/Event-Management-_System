# Event Manager Web Application

A modern event management platform with face recognition check-in, advanced dashboard, and admin controls.

## Features
- User registration, login, and authentication
- Event creation, editing, and deletion
- Dashboard with event search, filtering, and statistics
- Face recognition check-in for attendees (MediaPipe + OpenCV)
- Admin dashboard for user and event management
- **Admin profile management:** Admins can update their profile info and upload a profile picture directly from their profile window (`/admin/profile`).
- **"My Profile" button** on Admin dashboard for quick access.
- Notification system for users
- Responsive, modern UI with consistent branding
- Copyright footer centered on every page
- Calendar view (visible only to authenticated users)

## Tech Stack
- Python 3.x
- Flask (with Flask-Login, Flask-Mail)
- SQLite (default, can be swapped for other DBs)
- Pandas, scikit-learn (for analytics/ML)
- MediaPipe, OpenCV (for face recognition)
- Bootstrap 5 (CDN)

## Setup Instructions
1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd backend
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   > **Note:** For admin/user profile picture uploads, the `Pillow` library is required (see requirements.txt).
3. **Set up environment variables**
   - Copy `.env.example` to `.env` and configure as needed.
4. **Prepare static files**
   - Ensure the directory `static/profile_pics/` exists for profile pictures. Place a `default.png` image as the default avatar if desired.
5. **Run the application**
   ```bash
   python app.py
   ```
6. **Access the app**
   - Visit `http://localhost:5000` in your browser.

## File Structure
- `app.py` — Main Flask application
- `templates/` — HTML templates for all pages
- `static/` — CSS, JS, and image assets
- `utils/` — Face recognition and utility scripts
- `requirements.txt` — Python dependencies
- `.env.example` — Example environment config
- `README.md` — This file

## Deployment
- Designed for easy deployment to Heroku, Render, or similar PaaS.
- For production, use a WSGI server (e.g., Gunicorn) and configure environment variables securely.

## Credits
- Built with Flask, Bootstrap, MediaPipe, and OpenCV.

## License
MIT License
This project is licensed under the GPL license. See the LICENSES file for more details. 
---

**Demo Ready:**
- All major features tested and working
- UI is polished and consistent
- Face check-in is secure (real face required)
- Calendar is restricted to authenticated users
- Dashboard search is functional
