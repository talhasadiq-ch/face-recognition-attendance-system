"""
register_faces.py
------------------
Step 1 of the pipeline.

Captures face images of a new person from the webcam and stores them in
dataset/<person_name>/.

Computer vision concept used here: FACE DETECTION (not recognition yet).
We use OpenCV's Haar Cascade classifier to quickly find "is there a face
in this frame, and where". Detection just draws a box -- it has no idea
WHOSE face it is. Recognition (whose face) is a separate, later step.

Usage:
    python register_faces.py --name "John_Doe" --count 30
"""

import cv2
import os
import argparse


def register_faces(name: str, num_samples: int = 30, dataset_dir: str = "dataset"):
    person_dir = os.path.join(dataset_dir, name)
    os.makedirs(person_dir, exist_ok=True)

    # Haar Cascade: a classical, fast object-detection model shipped with OpenCV.
    # It works by scanning the image at multiple scales for patterns of light/dark
    # rectangles typical of eyes, nose bridge, etc. It's fast but less accurate
    # than deep-learning detectors -- fine for guiding capture, not for final recognition.
    face_detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)  # 0 = default webcam
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera index/permissions.")

    print(f"[INFO] Capturing {num_samples} images for '{name}'.")
    print("[INFO] Look at the camera. Press 'q' to quit early.")

    count = 0
    while count < num_samples:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # detection works on grayscale
        faces = face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,   # how much the image size is reduced at each scale
            minNeighbors=5,    # higher = fewer false positives, may miss real faces
            minSize=(80, 80),  # ignore tiny/far-away detections
        )

        for (x, y, w, h) in faces:
            count += 1
            face_img = frame[y:y + h, x:x + w]
            file_path = os.path.join(person_dir, f"{name}_{count}.jpg")
            cv2.imwrite(file_path, face_img)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Saved {count}/{num_samples}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            break  # only take one face per frame to avoid duplicate near-identical shots

        cv2.imshow("Registering Face - press q to stop", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Done. Saved {count} images to {person_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register a new person's face.")
    parser.add_argument("--name", required=True, help="Person's name (used as folder/label)")
    parser.add_argument("--count", type=int, default=30, help="Number of sample images to capture")
    args = parser.parse_args()

    register_faces(args.name.strip().replace(" ", "_"), args.count)
