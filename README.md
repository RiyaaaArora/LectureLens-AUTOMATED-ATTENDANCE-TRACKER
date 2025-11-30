Lecture Lens: Advanced Attendance System 🚀
Overview
Lecture Lens is an innovative, AI-powered attendance tracking system developed using Streamlit. It leverages face recognition and emotion detection to automate attendance marking during lectures, ensuring accuracy and efficiency. The system includes features for student registration, real-time attendance, timetable management, and an admin panel for data oversight.

Key Features ✨
Face Recognition: Uses OpenCV and face-recognition library for accurate student identification.
Emotion Detection: Integrates a Keras model to detect emotions (e.g., 😊 happy, 😢 sad) during attendance.
Smart Scheduling: Filters valid dates, excluding Sundays, 2nd/4th Saturdays, and public holidays.
Admin Panel: Secure login for managing students, faculty, and lectures.
Data Visualization: Pivoted attendance records with percentages and CSV export.
Modular Design: Code split into backend (logic) and frontend (UI) files for maintainability.


Installation 🛠️
Prerequisites
Python 3.8 or higher
Webcam access for face detection
Required models: CNN_Model_acc_75.h5 (emotion detection model)
Steps
Clone or Download: Get the project files into a directory.
Install Dependencies:
pip install streamlit opencv-python face-recognition pandas numpy pillow keras tensorflow
Note: face-recognition requires dlib. On Windows, install via pip install dlib or use pre-compiled wheels. Refer to face-recognition installation guide.
Setup Files:
Place CNN_Model_acc_75.h5 in the root directory.
Ensure Photos/ folder exists (auto-created if not).
The SQLite database (attendanc.db) will be created automatically on first run.
Run the App:
streamlit run main.py
Open the URL in your browser (e.g., http://localhost:8501).

Usage 📱
Getting Started
Home Dashboard: Select modes like Mark Attendance or Add New Face.
Modes Overview:
Mark Attendance: Capture face and emotion to mark attendance for the current lecture.
Add New Face: Register students with photos (checks for duplicates).
Close Attendance: Automatically mark absences after 15 minutes.
View Records: Browse pivoted attendance tables with exports.
View Attendance Summary: See daily/overall percentages.
View Timetable: Display lecture schedules.
Admin Panel: Login (ID: admin, Password: admin123) to manage data.

Tips 💡
Ensure camera permissions are enabled.
Attendance is only allowed within 10 minutes of lecture time on valid dates.
Use the sidebar for quick navigation.

File Structure 📂

.
├── config.py              # App configurations, styles, and session state
├── database.py            # SQLite database setup and queries
├── utils.py               # Helper functions (dates, lectures)
├── face_utils.py          # Face loading, encoding, and recognition
├── emotion_utils.py       # Emotion model loading and frame processing
├── ui_functions.py        # UI functions for attendance and views
├── admin.py               # Admin panel logic
├── main.py                # Main app runner
├── Photos/                # Directory for student face images
├── CNN_Model_acc_75.h5    # Pre-trained emotion detection model
└── attendanc.db           # SQLite database (auto-generated)

Dependencies 📦
streamlit: Web app framework
opencv-python: Computer vision
face-recognition: Face detection
pandas: Data handling
numpy: Arrays and math
pillow: Image processing
keras & tensorflow: Deep learning for emotions
sqlite3: Database (Python built-in)

Configuration ⚙️
Admin Credentials: Change in admin.py (default: admin/admin123).
Holidays/Dates: Edit public_holidays and date ranges in utils.py.
Lectures/Faculty: Add via Admin Panel or directly in database.py.
Model Path: Ensure CNN_Model_acc_75.h5 is accessible.


Troubleshooting 🔧
Camera Issues: Check browser permissions.
Model Errors: Verify model file and Keras/TensorFlow versions.
Database: If corrupted, delete attendanc.db to recreate.
Imports: Run in a virtual environment to avoid conflicts.


Contributing 🤝
Contributions are welcome! Fork the repo, make changes, and submit a PR. Keep functions unchanged as per original design. Focus on bug fixes, UI improvements, or new features.

License 📄
This project is for educational purposes. Adapt and use responsibly. No formal license applied.

For questions or support, check Streamlit docs or raise an issue. Enjoy using Lecture Lens! 🚀
