# Face Attendance System

#Updated README for lab submission


A minimal face-based attendance system using **OpenCV** and **Streamlit**.

---

## 🧠 Features

- **Face Registration**  
  Capture and save a new face with a name using the webcam.

- **Attendance Marking**  
  Detect and recognize registered faces to log attendance automatically.

- **Attendance History**  
  - View all previous attendance entries  
  - Filter by name  
  - Download CSV for records  
  - Basic statistics: total entries, unique people  

- **Lightweight Recognition**  
  Uses **OpenCV Haar Cascades** and simple grayscale embeddings (no heavy ML models).

- **Streamlit Deployment**  
  Simple web UI for registration, attendance, and history.

---

## 💻 Technologies Used

- Python 3.x  
- OpenCV  
- Streamlit  
- NumPy  
- Pandas  
- PIL (Python Imaging Library)
- exception handling
- OOP

---

## 🚀 How to Run

1. Clone or download the repository.
2. Create a virtual environment and activate it:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt

python -m streamlit run "Face Detection Algorithm/app.py"

copy/
├─ Face Detection Algorithm/   # Streamlit app code
├─ Images/                     # Registered face images
├─ Attendance.csv              # Attendance log
├─ README.md                   # Project documentation
├─ requirements.txt            # Dependencies
└─ utils_plots_main.py         # Helper scripts

