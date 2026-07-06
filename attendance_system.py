"""
attendance_system.py
---------------------
Step 3 of the pipeline -- the live system.

Pipeline per video frame:
    1. DETECT   -- find where faces are (face_recognition.face_locations)
    2. ENCODE   -- turn each detected face into a 128-d vector
    3. COMPARE  -- measure distance to every known encoding
    4. DECIDE   -- closest match under a distance threshold -> recognized person
                   otherwise -> "Unknown"
    5. LOG      -- if recognized and not already marked today, append a row
                   to attendance/attendance_<date>.csv with a timestamp

Run:
    python attendance_system.py
Press 'q' to quit.
"""

import face_recognition
import cv2
import pickle
import os
import csv
from datetime import datetime

ENCODINGS_PATH = "encodings/encodings.pickle"
ATTENDANCE_DIR = "attendance"
# Lower = stricter match (fewer false positives, may reject valid faces).
# 0.6 is the widely-used default for the dlib model this library wraps.
MATCH_TOLERANCE = 0.5
# Skip every Nth frame's detection cost by resizing -- speeds up recognition
# significantly with a small accuracy trade-off.
RESIZE_SCALE = 0.25


def load_encodings(path: str = ENCODINGS_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run register_faces.py then encode_faces.py first."
        )
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["encodings"], data["names"]


def get_today_csv_path():
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")


def load_already_marked(csv_path: str):
    marked = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if row:
                    marked.add(row[0])
    return marked


def mark_attendance(name: str, csv_path: str, marked_today: set):
    if name in marked_today:
        return  # already logged once today -- don't spam the CSV every frame
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Date", "Time"])
        now = datetime.now()
        writer.writerow([name, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")])
    marked_today.add(name)
    print(f"[ATTENDANCE] Marked {name} at {datetime.now().strftime('%H:%M:%S')}")


def run():
    known_encodings, known_names = load_encodings()
    csv_path = get_today_csv_path()
    marked_today = load_already_marked(csv_path)

    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("[INFO] System running. Press 'q' to quit.")

    while True:
        ok, frame = video_capture.read()
        if not ok:
            break

        # Shrink frame before the expensive detection/encoding step, then scale
        # coordinates back up when drawing -- a common real-time CV optimization.
        small_frame = cv2.resize(frame, (0, 0), fx=RESIZE_SCALE, fy=RESIZE_SCALE)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            name = "Unknown"

            if len(distances) > 0:
                best_match_index = distances.argmin()
                if distances[best_match_index] <= MATCH_TOLERANCE:
                    name = known_names[best_match_index]
                    mark_attendance(name, csv_path, marked_today)

            # Scale face box back up to original frame size
            top = int(top / RESIZE_SCALE)
            right = int(right / RESIZE_SCALE)
            bottom = int(bottom / RESIZE_SCALE)
            left = int(left / RESIZE_SCALE)

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Face Attendance System - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Session ended. Attendance saved to {csv_path}")


if __name__ == "__main__":
    run()
