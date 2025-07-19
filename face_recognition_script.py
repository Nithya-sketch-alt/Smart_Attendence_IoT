import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime, time as dtime
import firebase_admin
from firebase_admin import credentials, db

# 🔐 Firebase setup
cred = credentials.Certificate("classroom-attendence-firebase-adminsdk-fbsvc-3821a5e84f.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://classroom-attendence-default-rtdb.firebaseio.com/'
    })

# 📁 Known Faces Directory
KNOWN_FACES_BASE_DIR = 'known_faces'
current_class = input("Enter the class to monitor (e.g., class A): ").strip()
class_path = os.path.join(KNOWN_FACES_BASE_DIR, current_class)
if not os.path.exists(class_path):
    print(f"❌ Class folder '{current_class}' not found.")
    exit()

# 📥 Load face encodings
known_encodings = []
known_names = []
for filename in os.listdir(class_path):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        path = os.path.join(class_path, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0]
            known_names.append(name)

# ✅ Upload student list
students_ref = db.reference(f"students/{current_class}")
if not students_ref.get():
    students_ref.set(known_names)

# 🎥 Webcam
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("❌ Unable to access the camera.")
    exit()

# ⚙ Setup
FACE_MATCH_THRESHOLD = 0.5
attendance_sent = set()
present_students = {}
today_date = datetime.now().strftime("%Y-%m-%d")
attendance_path = f"attendance/{current_class}/{today_date}"

# 🎯 Time Window: 9:00 AM to 12:30 PM
start_time_limit = dtime(hour=9, minute=0)
end_time_limit = dtime(hour=12, minute=30)

while True:
    now = datetime.now()
    current_time_only = now.time()

    if start_time_limit <= current_time_only <= end_time_limit:
        ret, frame = video_capture.read()
        if not ret:
            continue

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
            name = "Unknown"
            if known_encodings:
                face_distances = face_recognition.face_distance(known_encodings, encoding)
                best_match_index = np.argmin(face_distances)

                if face_distances[best_match_index] < FACE_MATCH_THRESHOLD:
                    name = known_names[best_match_index]
                    if name not in attendance_sent:
                        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                        attendance_ref = db.reference(f"{attendance_path}/{name}")
                        attendance_ref.set({
                            "timestamp": timestamp,
                            "status": "Present"
                        })
                        print(f"✅ Marked '{name}' Present at {timestamp}")
                        attendance_sent.add(name)
                        present_students[name] = timestamp

                        # 🧾 Summary update
                        total_students = len(known_names)
                        total_present = len(present_students)
                        total_absent = total_students - total_present

                        absent_students = list(set(known_names) - set(present_students.keys()))

                        summary_ref = db.reference(f"{attendance_path}/summary")
                        summary_ref.set({
                            "total_students": total_students,
                            "total_present": total_present,
                            "total_absent": total_absent,
                            "present_students": present_students,
                            "absent_students": absent_students,
                            "timestamp": timestamp
                        })

            # 🖼 Draw on screen
            top *= 2; right *= 2; bottom *= 2; left *= 2
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, bottom + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # ℹ Info
        cv2.putText(frame, f"Class: {current_class}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Present: {len(present_students)} / {len(known_names)}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 0), 2)
        cv2.putText(frame, f"Time: {now.strftime('%H:%M:%S')}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

        cv2.imshow("Face Recognition Attendance", frame)
    else:
        # ⏱ Outside time window
        frame = np.zeros((400, 800, 3), dtype=np.uint8)
        msg = "Attendance allowed only from 9:00 AM to 12:30 PM"
        cv2.putText(frame, msg, (20, 200), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 255), 2)
        cv2.imshow("Face Recognition Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Exiting program.")
        break

# 🧹 Cleanup
video_capture.release()
cv2.destroyAllWindows()
