import os
import cv2
import time
import numpy as np
from keras._tf_keras.keras.models import load_model

# Load face detector and emotion classifier
face_classifier = cv2.CascadeClassifier(
    r'C:\Users\User\Documents\Year 2 (Sem 3)\Methods and Applications of Deep Learning\Final Project\Final Project\haarcascade_frontalface_default.xml'
)
classifier = load_model(
    r'C:\Users\User\Documents\Year 2 (Sem 3)\Methods and Applications of Deep Learning\Final Project\Final Project\models\CNN_best.keras'
)
class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Test image folder path
test_folder = r'C:\Users\User\Documents\Year 2 (Sem 3)\Methods and Applications of Deep Learning\Final Project\Final Project\testing_images'

def preprocess_face(roi_rgb):
    roi_rgb = cv2.resize(roi_rgb, (48, 48), interpolation=cv2.INTER_AREA)
    roi_rgb = roi_rgb.astype("float32") / 255.0
    roi_rgb = np.expand_dims(roi_rgb, axis=0)
    return roi_rgb

def predict_and_annotate(frame, faces, show=True):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    for (x, y, w, h) in faces:
        roi = preprocess_face(rgb_frame[y:y+h, x:x+w])
        preds = classifier.predict(roi)[0]
        label = class_labels[preds.argmax()]
        confidence = preds.max()
        text = f"{label} ({confidence*100:.1f}%)"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    if show:
        cv2.imshow("Emotion Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def webcam_mode():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

        predict_and_annotate(frame, faces, show=False)
        cv2.imshow("Real-Time Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def capture_and_test_image():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Press ENTER to Capture / ESC to Exit", cv2.WINDOW_NORMAL)
    print("[INFO] Webcam is on. Press ENTER to capture, ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            return

        cv2.imshow("Press ENTER to Capture / ESC to Exit", frame)
        key = cv2.waitKey(1)

        if key % 256 == 13:  # Enter key
            timestamp = int(time.time())
            filename = f"captured_{timestamp}.jpg"
            save_path = os.path.join(test_folder, filename)
            cv2.imwrite(save_path, frame)
            print(f"[INFO] Image saved to {save_path}")
            break
        elif key % 256 == 27:  # Esc key
            print("[INFO] Capture cancelled.")
            return

    cap.release()
    cv2.destroyAllWindows()

    image = cv2.imread(save_path)
    if image is None:
        print("Error loading captured image.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        print("No faces detected.")
    else:
        predict_and_annotate(image, faces)

def test_existing_images():
    for filename in os.listdir(test_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(test_folder, filename)
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to load {filename}")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.1, 4)

            if len(faces) == 0:
                print(f"No faces found in {filename}")
            else:
                print(f"{filename} - Faces detected: {len(faces)}")
                predict_and_annotate(image, faces)

# === MAIN MENU ===
print("Select Emotion Detection Mode:")
print("1. Real-time Webcam Detection")
print("2. Capture and Analyze a New Image")
print("3. Analyze Existing Images in Folder")

choice = input("Enter your choice (1/2/3): ")

if choice == '1':
    webcam_mode()
elif choice == '2':
    capture_and_test_image()
elif choice == '3':
    test_existing_images()
else:
    print("Invalid choice.")
