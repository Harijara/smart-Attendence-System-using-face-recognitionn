import tkinter as tk
from tkinter import messagebox
import cv2
import face_recognition
import pandas as pd
import numpy as np
from datetime import datetime
import os
print("📂 Current directory:", os.getcwd())
from PIL import Image, ImageTk
import threading

# === Global Variables ===
running = False
known_face_encodings = []
known_face_names = []
recognized_this_session = set()
video_capture = None

# === Load Dataset ===
def load_dataset():
    global known_face_encodings, known_face_names
    known_face_encodings.clear()
    known_face_names.clear()

    if not os.path.exists("Dataset2.csv"):
        return

    df = pd.read_csv("Dataset2.csv")
    for index, row in df.iterrows():
        path = row['ImagePath']
        if os.path.exists(path):
            img = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(img)
            if encodings:
                known_face_encodings.append(encodings[0])
                known_face_names.append(f"{row['RegdNo']} - {row['Name']}")

# === Attendance Logger ===
def mark_attendance(name):
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open("attendance_output.csv", "a") as f:
        f.write(f"{name},{now}\n")
        print(f"📝 Wrote to CSV: {name},{now}")

# === Face Recognition Logic ===
def recognize_faces():
    global running, video_capture, recognized_this_session
    print("🟢 Recognition thread started")
    recognized_this_session.clear()
    load_dataset()
    print("✅ Dataset loaded:", known_face_names)

    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("❌ ERROR: Cannot open webcam.")
        return

    print("📷 Webcam opened")

    while running:
        ret, frame = video_capture.read()
        if not ret:
            print("⚠️ Failed to read frame")
            continue

        print("✅ Frame captured")

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        print(f"Detected {len(face_encodings)} face(s)")

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            name = "Unknown"

           # Encode & Match
            matches = face_recognition.compare_faces(known_face_encodings, encoding)
            face_distances = face_recognition.face_distance(known_face_encodings, encoding)

            if matches and any(matches):
                best_match = np.argmin(face_distances)
                if matches[best_match]:
                   name = known_face_names[best_match]
            else:
    # Generate stable ID for unknown person using part of encoding
                unknown_id = str(hash(tuple(encoding[:5])))[:6]
                name = f"Unknown_{unknown_id}"

# Ensure we log this unique face once per run
            if name not in recognized_this_session:
                recognized_this_session.add(name)
                mark_attendance(name)
                print(f"✅ Marked attendance: {name}")


            top *= 4; right *= 4; bottom *= 4; left *= 4
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            print(f"🧪 Found face named: {name}")
            

            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            print("🚪 ESC key pressed. Stopping.")
            break

        if not running:
            print("🛑 Recognition stopped via button.")
            break

    video_capture.release()
    cv2.destroyAllWindows()
    print("🔴 Webcam released")


# === Start & Stop Buttons ===
def start_recognition():
    global running
    if not running:
        running = True
        threading.Thread(target=recognize_faces, daemon=True).start()

def stop_recognition():
    global running
    running = False

# === Add New Person ===
def add_person():
    regd = entry_id.get()
    name = entry_name.get()

    if not regd or not name:
        messagebox.showerror("Error", "Enter ID and Name")
        return

    save_path = f"dataset/{name.lower()}_{regd}.jpg"

    cap = cv2.VideoCapture(0)
    messagebox.showinfo("Capture", "Press SPACE to capture image, ESC to cancel")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Capture Face", frame)
        key = cv2.waitKey(1)

        if key % 256 == 27:  # ESC
            break
        elif key % 256 == 32:  # SPACE
            cv2.imwrite(save_path, frame)
            break

    cap.release()
    cv2.destroyAllWindows()

    # Fix for pandas ≥ 2.0
    df = pd.read_csv("Dataset2.csv") if os.path.exists("Dataset2.csv") else pd.DataFrame(columns=["RegdNo", "Name", "ImagePath"])
    new_row = pd.DataFrame([{"RegdNo": regd, "Name": name, "ImagePath": save_path}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv("Dataset2.csv", index=False)
    messagebox.showinfo("Success", f"{name} added!")
    load_dataset()

# === GUI Layout ===
root = tk.Tk()
root.title("Face Recognition Attendance")
root.geometry("400x400")

tk.Label(root, text="Face Recognition System", font=("Arial", 16)).pack(pady=10)

btn_start = tk.Button(root, text="Start Recognition", font=("Arial", 12), command=start_recognition)
btn_start.pack(pady=10)

btn_stop = tk.Button(root, text="Stop Recognition", font=("Arial", 12), command=stop_recognition)
btn_stop.pack(pady=10)

tk.Label(root, text="Add New Person", font=("Arial", 14)).pack(pady=20)
tk.Label(root, text="Regd No:").pack()
entry_id = tk.Entry(root)
entry_id.pack()

tk.Label(root, text="Name:").pack()
entry_name = tk.Entry(root)
entry_name.pack()

btn_add = tk.Button(root, text="Add Person", font=("Arial", 12), command=add_person)
btn_add.pack(pady=10)

root.mainloop()
