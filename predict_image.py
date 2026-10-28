
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

image_path = input("Enter the image path: ")

image = cv2.imread(image_path)

if image is None:
    print("Could not load the image. Check the file path.")
else:
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4
    )

    if len(faces) == 0:
        print("No face detected.")
    else:
        x, y, w, h = faces[0]

        face = image[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = np.expand_dims(face, axis=0)
        face = face.astype("float32")

        prediction = new_model.predict(face, verbose=0)

        index = int(np.argmax(prediction[0]))

        print("Predicted emotion:", Classes[index])

        for i, emotion in enumerate(Classes):
            score = float(prediction[0][i]) * 100
            print(emotion, round(score, 2), "%")

        cv2.rectangle(
            image,
            (x, y),
            (x+w, y+h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            image,
            Classes[index].capitalize(),
            (x, max(y-10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Emotion Prediction", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
