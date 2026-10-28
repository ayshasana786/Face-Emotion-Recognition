
import os
import cv2
import numpy as np
import tensorflow as tf

Classes = [
    "surprise",
    "sad",
    "neutral",
    "happy",
    "fear",
    "disgust",
    "angry"
]

model_path = os.path.join(
    os.path.dirname(__file__),
    "models",
    "Final_model_95p07.h5"
)

new_model = tf.keras.models.load_model(model_path)

faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

if faceCascade.empty():
    raise IOError("Face detector could not be loaded")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    cap = cv2.VideoCapture(1)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

try:
    while True:
        ret, frame = cap.read()

        if not ret:
            print("Cannot read webcam frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=4
        )

        for x, y, w, h in faces:
            face_roi = frame[y:y+h, x:x+w]

            final_image = cv2.resize(
                face_roi, (224, 224)
            )

            final_image = np.expand_dims(
                final_image, axis=0
            )

            final_image = final_image.astype("float32")

            predictions = new_model.predict(
                final_image, verbose=0
            )

            index = int(np.argmax(predictions[0]))
            status = Classes[index]

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                status.capitalize(),
                (x, max(y-12, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.imshow("Real-Time Emotion Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
