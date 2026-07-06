"""
app.py
------
Streamlit dashboard for the Face Attendance System.

Replaces running register_faces.py / encode_faces.py / attendance_system.py
separately from the terminal -- everything is now one browser dashboard.

Run with:
    streamlit run app.py

Four sections (see sidebar):
    1. Register       -- capture face photos for a new person
    2. Build Encodings -- convert saved photos into the recognition database
    3. Live Attendance -- real-time detect + recognize + auto-mark attendance
    4. Records         -- view / download past attendance CSVs
"""

import streamlit as st
import cv2
import os
import csv
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

import face_recognition

# ---------------------------------------------------------------------------
# Paths / constants (same layout as the original CLI scripts)
# ---------------------------------------------------------------------------
DATASET_DIR = "dataset"
ENCODINGS_PATH = "encodings/encodings.pickle"
ATTENDANCE_DIR = "attendance"
MATCH_TOLERANCE = 0.5

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs("encodings", exist_ok=True)
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

st.set_page_config(page_title="Face Attendance System", page_icon="🧑‍💻", layout="wide")


# ---------------------------------------------------------------------------
# Shared helper functions
# ---------------------------------------------------------------------------
def get_registered_people():
    return sorted([
        p for p in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, p))
    ])


def count_photos(name):
    person_dir = os.path.join(DATASET_DIR, name)
    if not os.path.isdir(person_dir):
        return 0
    return len([f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])


def build_encodings():
    """Same logic as encode_faces.py, callable from the dashboard."""
    known_encodings, known_names = [], []
    people = get_registered_people()

    progress = st.progress(0.0, text="Starting...")
    for i, person_name in enumerate(people):
        person_dir = os.path.join(DATASET_DIR, person_name)
        image_files = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        for image_file in image_files:
            image = cv2.imread(os.path.join(person_dir, image_file))
            if image is None:
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb, model="hog")
            encs = face_recognition.face_encodings(rgb, boxes)
            for enc in encs:
                known_encodings.append(enc)
                known_names.append(person_name)

        progress.progress((i + 1) / max(len(people), 1), text=f"Processed {person_name}")

    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump({"encodings": known_encodings, "names": known_names}, f)

    progress.empty()
    return len(known_encodings), len(set(known_names))


@st.cache_resource(show_spinner=False)
def load_encodings(_mtime):
    """_mtime is passed so Streamlit's cache auto-invalidates when the file changes."""
    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]


def get_today_csv_path():
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")


def load_already_marked(csv_path):
    marked = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    marked.add(row[0])
    return marked


def mark_attendance(name, csv_path, marked_today):
    if name in marked_today:
        return False
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Date", "Time"])
        now = datetime.now()
        writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
    marked_today.add(name)
    return True


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🧑‍💻 Face Attendance")
page = st.sidebar.radio(
    "Go to",
    ["1. Register Person", "2. Build Encodings", "3. Live Attendance", "4. Attendance Records"],
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Registered people: {len(get_registered_people())}")
st.sidebar.caption(f"Encodings file: {'✅ found' if os.path.exists(ENCODINGS_PATH) else '❌ not built yet'}")


# ---------------------------------------------------------------------------
# PAGE 1: Register a new person (browser camera snapshot, click-to-save)
# ---------------------------------------------------------------------------
if page == "1. Register Person":
    st.header("Register a New Person")
    st.write("Take a few photos from slightly different angles (front, left, right, "
             "with/without glasses if applicable). 10-20 photos gives solid accuracy.")

    name = st.text_input("Person's name").strip().replace(" ", "_")

    if name:
        st.caption(f"Photos saved so far for **{name}**: {count_photos(name)}")

        snapshot = st.camera_input("Take a photo", key=f"cam_{name}")

        if snapshot is not None:
            file_bytes = np.asarray(bytearray(snapshot.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

            if len(faces) == 0:
                st.warning("No face detected in that photo -- try again with better lighting/angle.")
            else:
                x, y, w, h = faces[0]
                face_crop = frame[y:y + h, x:x + w]

                person_dir = os.path.join(DATASET_DIR, name)
                os.makedirs(person_dir, exist_ok=True)
                next_index = count_photos(name) + 1
                save_path = os.path.join(person_dir, f"{name}_{next_index}.jpg")
                cv2.imwrite(save_path, face_crop)

                st.success(f"Saved photo #{next_index} for {name}. Take another, or move to step 2.")
                st.image(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB), width=200, caption="Saved crop")
    else:
        st.info("Enter a name above to start capturing photos.")

    st.markdown("---")
    st.subheader("Currently registered")
    people = get_registered_people()
    if people:
        st.table(pd.DataFrame({"Name": people, "Photos": [count_photos(p) for p in people]}))
    else:
        st.caption("No one registered yet.")


# ---------------------------------------------------------------------------
# PAGE 2: Build / rebuild encodings
# ---------------------------------------------------------------------------
elif page == "2. Build Encodings":
    st.header("Build Face Encodings")
    st.write("This converts every saved photo into a 128-dimensional face encoding "
             "and saves the recognition database. Run this any time you register someone new.")

    people = get_registered_people()
    if not people:
        st.warning("No registered people yet -- go to 'Register Person' first.")
    else:
        st.write(f"Found **{len(people)}** people, **{sum(count_photos(p) for p in people)}** total photos.")
        if st.button("🔄 Build / Rebuild Encodings", type="primary"):
            with st.spinner("Encoding faces... this can take a minute."):
                n_encodings, n_people = build_encodings()
            st.success(f"Done! Encoded {n_encodings} face samples across {n_people} people.")
            st.cache_resource.clear()  # force live attendance page to reload fresh encodings


# ---------------------------------------------------------------------------
# PAGE 3: Live attendance monitoring
# ---------------------------------------------------------------------------
elif page == "3. Live Attendance":
    st.header("Live Attendance")

    if not os.path.exists(ENCODINGS_PATH):
        st.error("No encodings found yet. Go to 'Build Encodings' first.")
    else:
        mtime = os.path.getmtime(ENCODINGS_PATH)
        known_encodings, known_names = load_encodings(mtime)
        st.caption(f"Loaded {len(known_encodings)} encodings for {len(set(known_names))} people.")

        csv_path = get_today_csv_path()
        if "marked_today" not in st.session_state:
            st.session_state.marked_today = load_already_marked(csv_path)

        run = st.checkbox("▶️ Start camera")
        frame_window = st.image([])
        status_placeholder = st.empty()

        if run:
            cap = cv2.VideoCapture(0)
            RESIZE_SCALE = 0.25
            while run:
                ok, frame = cap.read()
                if not ok:
                    st.error("Could not read from webcam.")
                    break

                small = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
                rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

                locations = face_recognition.face_locations(rgb_small, model="hog")
                encodings = face_recognition.face_encodings(rgb_small, locations)

                for (top, right, bottom, left), enc in zip(locations, encodings):
                    distances = face_recognition.face_distance(known_encodings, enc)
                    name = "Unknown"
                    if len(distances) > 0:
                        best = distances.argmin()
                        if distances[best] <= MATCH_TOLERANCE:
                            name = known_names[best]
                            if mark_attendance(name, csv_path, st.session_state.marked_today):
                                status_placeholder.success(f"Marked present: {name} at {datetime.now().strftime('%H:%M:%S')}")

                    top, right, bottom, left = [int(v / RESIZE_SCALE) for v in (top, right, bottom, left)]
                    color = (0, 200, 0) if name != "Unknown" else (0, 0, 220)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            cap.release()
        else:
            st.caption("Camera is off. Check the box above to start recognizing and marking attendance.")


# ---------------------------------------------------------------------------
# PAGE 4: Records viewer
# ---------------------------------------------------------------------------
elif page == "4. Attendance Records":
    st.header("Attendance Records")

    files = sorted(
        [f for f in os.listdir(ATTENDANCE_DIR) if f.endswith(".csv")],
        reverse=True,
    )

    if not files:
        st.info("No attendance recorded yet.")
    else:
        selected = st.selectbox("Select date", files)
        df = pd.read_csv(os.path.join(ATTENDANCE_DIR, selected))
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "⬇️ Download CSV",
            data=df.to_csv(index=False),
            file_name=selected,
            mime="text/csv",
        )
        st.caption(f"{len(df)} people marked present on this date.")
