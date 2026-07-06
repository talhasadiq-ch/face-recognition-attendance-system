# Face Recognition Attendance System

A real, working attendance system using **Python + OpenCV + face_recognition**,
controlled through a **Streamlit dashboard** (no manual commands needed once it's running).

---

## 📁 Project structure

```
face_attendance/
├── app.py                 <- the dashboard (run this)
├── register_faces.py      <- optional: same task via CLI
├── encode_faces.py         <- optional: same task via CLI
├── attendance_system.py    <- optional: same task via CLI
├── requirements.txt
├── dataset/                <- created automatically: photos per person
├── encodings/               <- created automatically: encodings.pickle
├── attendance/               <- created automatically: daily CSV logs
└── .vscode/
    ├── settings.json        <- points VS Code to the project's venv
    └── launch.json           <- lets you run app.py with F5 (debugger)
```

The three CLI scripts (`register_faces.py`, `encode_faces.py`, `attendance_system.py`)
still work standalone if you ever want to script things outside the browser — but
day-to-day you'll just use `app.py`.

---

## 🖥️ Full guide: opening and running this in VS Code

### Step 1 — Install prerequisites (once)
- [VS Code](https://code.visualstudio.com/)
- [Python 3.10+](https://www.python.org/downloads/) — during install on Windows, **check "Add Python to PATH"**
- In VS Code: go to the Extensions panel (left sidebar, square-icon) and install:
  - **Python** (by Microsoft)
  - **Pylance** (usually installs automatically with Python)

### Step 2 — Open the project folder
- `File → Open Folder...` → select the `face_attendance` folder (the one with `app.py` in it).
- VS Code will show the file tree on the left — you should see `app.py`, `dataset/`, etc.

### Step 3 — Open a terminal inside VS Code
- `Terminal → New Terminal` (or `` Ctrl+` ``). This opens a terminal already pointed at your project folder.

### Step 4 — Create a virtual environment (keeps this project's packages isolated)
```bash
python -m venv venv
```
Activate it:
- **Windows (PowerShell)**: `venv\Scripts\Activate.ps1`
- **Mac/Linux**: `source venv/bin/activate`

You'll know it worked when your terminal prompt shows `(venv)` at the start.

> If VS Code pops up a message "Select Python Interpreter" or similar, click it and
> choose the one inside `venv` (e.g. `./venv/bin/python` or `.\venv\Scripts\python.exe`).
> This is also configured for you already in `.vscode/settings.json`.

### Step 5 — Install dependencies
```bash
pip install -r requirements.txt
```
`dlib` (needed by `face_recognition`) compiles from source the first time, so:
- **Windows**: install [CMake](https://cmake.org/download/) and "Desktop development with C++"
  (from Visual Studio Build Tools) before this step, or it will fail.
- **Mac**: `brew install cmake` first.
- **Linux**: `sudo apt install build-essential cmake` first.

This step can take a few minutes the first time — that's normal.

### Step 6 — Run the dashboard
Two ways:
- **Terminal**: `streamlit run app.py` — it prints a `Local URL` (usually `http://localhost:8501`)
  and should open automatically in your browser.
- **VS Code debugger**: press `F5` and pick "Streamlit: app.py" (already configured in `launch.json`) —
  lets you set breakpoints in `app.py` if you want to step through the code.

### Step 7 — Use the dashboard
The sidebar has 4 pages, meant to be used in order:
1. **Register Person** — type a name, click the camera widget to snapshot a photo,
   repeat 10-20 times from different angles. Do this once per person.
2. **Build Encodings** — click the button. This processes everyone's photos into the
   recognition database (`encodings/encodings.pickle`). Re-run this any time you add someone.
3. **Live Attendance** — check "Start camera". It'll draw a green box + name over
   recognized people and log them once per day; unknown faces get a red box.
4. **Attendance Records** — pick a date, view/download the CSV log for that day.

### Stopping the app
In the terminal running Streamlit, press `Ctrl+C`. To leave the virtual environment
afterward, type `deactivate`.

---

## How it works — the underlying pipeline

```
Register (detect + save)  →  Build Encodings (recognize → 128-d vector)  →  Live Attendance (compare + log)
```

- **Detection** (Haar Cascade): "is there a face, and where" — used when saving registration photos.
- **Recognition** (`face_recognition` / dlib ResNet): turns a face into a 128-dimensional
  encoding. Same person's faces cluster close together in that space; different people are far apart.
- **Matching**: at attendance time, a live face's encoding is compared by distance to every
  known encoding — closest one under `MATCH_TOLERANCE` (0.5) wins; otherwise "Unknown".
- **Attendance logging**: first recognition of a person each day writes a row
  (Name, Date, Time) to `attendance/attendance_<date>.csv`; later detections that day are skipped.

## Key computer vision concepts you'll now understand

| Concept | Where it shows up | Why it matters |
|---|---|---|
| **Detection vs. Recognition** | Haar Cascade (detect) vs. face_recognition (recognize) | Detection = "a face exists here." Recognition = "whose face." Different problems, different models. |
| **Face encodings / embeddings** | `build_encodings()` in `app.py` | Faces are compared as 128-d vectors, not raw pixels — robust to lighting/angle changes. |
| **Distance threshold** | `MATCH_TOLERANCE = 0.5` | Recognition is never 100% certain; you're picking a cutoff that trades off false accepts vs. false rejects. |
| **HOG vs CNN detector** | `model="hog"` param | HOG = fast, CPU-friendly, slightly less accurate. CNN = more accurate, needs GPU for real-time speed. |
| **Frame downscaling** | `RESIZE_SCALE = 0.25` | Real-time CV constantly trades resolution for speed — detect on a small frame, draw on the full one. |
| **Color space conversion** | `BGR2RGB`, `BGR2GRAY` | OpenCV loads images as BGR; most CV/ML libraries expect RGB or grayscale — an easy bug source. |
| **State/deduplication logic** | `st.session_state.marked_today` | Real systems need business logic layered on top of raw CV output (don't log the same person 30 times/second). |

## Practical next steps to extend this project
- Store attendance in a proper database (SQLite/PostgreSQL) instead of CSV.
- Add liveness detection (blink detection) to prevent someone holding up a photo.
- Swap Haar Cascade for a more modern detector (e.g. MTCNN or a YOLO-face model) for better accuracy in low light/angles.
- Add email/Slack notification when someone is marked present.
- Add authentication to the dashboard if this will be used by more than just you (Streamlit supports this via `streamlit-authenticator`).

## Troubleshooting
- **"Could not open webcam"** — another app (Zoom, Teams) may be using the camera; close it and retry.
- **dlib install fails** — see the CMake/Build Tools note in Step 5 above.
- **Camera works in the browser widget (Register page) but not in Live Attendance** — the Register
  page uses your browser's camera via `st.camera_input`; Live Attendance uses OpenCV's direct camera
  access (`cv2.VideoCapture(0)`) on the machine running Streamlit. If you're running Streamlit locally,
  these are the same camera; if you ever deploy this to a remote server, `cv2.VideoCapture(0)` would
  access the *server's* camera, not yours — fine for local use, but worth knowing.
