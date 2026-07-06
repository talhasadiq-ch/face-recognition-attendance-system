"""
encode_faces.py
----------------
Step 2 of the pipeline.

Reads every image in dataset/<person_name>/*.jpg and converts each face into
a 128-dimensional numeric vector called a FACE ENCODING (or "face embedding").

Computer vision concept: FACE RECOGNITION via embeddings.
Instead of comparing raw pixels (which changes with lighting/angle/expression),
a deep neural network (a ResNet trained on millions of faces, bundled inside
the `face_recognition` / dlib library) maps a face to a point in 128-dimensional
space such that:
    - Same person's faces  -> points close together
    - Different people     -> points far apart

Later, recognizing someone is just: "find the closest known point to this new
face's point." That's why this step must run once whenever you add a new person.

Output: encodings/encodings.pickle
    { "encodings": [vec1, vec2, ...], "names": ["John_Doe", "John_Doe", ...] }
"""

import face_recognition
import os
import pickle
import cv2


def encode_faces(dataset_dir: str = "dataset", output_path: str = "encodings/encodings.pickle"):
    known_encodings = []
    known_names = []

    people = [p for p in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, p))]
    if not people:
        print("[WARN] No person folders found in dataset/. Run register_faces.py first.")
        return

    for person_name in people:
        person_dir = os.path.join(dataset_dir, person_name)
        image_files = [f for f in os.listdir(person_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
        print(f"[INFO] Processing {person_name}: {len(image_files)} images")

        for image_file in image_files:
            image_path = os.path.join(person_dir, image_file)
            image = cv2.imread(image_path)
            if image is None:
                continue
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # face_recognition expects RGB, OpenCV loads BGR

            # Locate face(s) in the image, then compute the 128-d encoding for each.
            # model="hog" is CPU-friendly and fast; model="cnn" is more accurate but needs a GPU to be fast.
            boxes = face_recognition.face_locations(rgb, model="hog")
            encodings = face_recognition.face_encodings(rgb, boxes)

            for enc in encodings:
                known_encodings.append(enc)
                known_names.append(person_name)

    data = {"encodings": known_encodings, "names": known_names}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(data, f)

    print(f"[INFO] Saved {len(known_encodings)} encodings for {len(set(known_names))} people to {output_path}")


if __name__ == "__main__":
    encode_faces()
