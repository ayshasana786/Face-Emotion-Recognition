# Real-Time Face Emotion Recognition

A deep learning and computer vision project that predicts facial-expression categories from images and live webcam video using the FER-2013 dataset.

## 1. Project Overview

This project implements a facial emotion recognition system using a trained deep learning model, TensorFlow, Keras, and OpenCV.

The application detects a face in an image or webcam frame, preprocesses the detected face, and predicts one of seven facial-expression categories.

The project supports single-image prediction through a Python script or Jupyter Notebook and real-time emotion prediction through a webcam.

## 2. Dataset

This project uses the **FER-2013 facial expression dataset** from Kaggle.

Dataset: [FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)

The dataset contains facial images organized into seven emotion categories:

- Surprise
- Sad
- Neutral
- Happy
- Fear
- Disgust
- Angry

The dataset is used locally and is not included in this repository.

## 3. Technologies Used

- Python
- TensorFlow and Keras
- OpenCV
- NumPy
- Matplotlib
- Jupyter Notebook
- Haar Cascade face detection

## 4. Project Structure

```text
Face-Emotion-Recognition/
├── models/
│   └── Final_model_95p07.h5
├── face_emotion_recognition.ipynb
├── predict_image.py
├── real_time_emotion.py
├── requirements.txt
├── README.md
└── .gitignore
```

The dataset, virtual environment, and personal images are kept locally and excluded from version control.

## 5. Project Workflow

1. Load and explore the FER-2013 dataset.
2. Read images and assign emotion labels.
3. Preprocess images for model input.
4. Build the deep learning model.
5. Train the model on the prepared dataset.
6. Save the trained model.
7. Load the saved model for inference.
8. Detect faces using OpenCV's Haar Cascade detector.
9. Predict facial-expression categories from images or webcam frames.

## 6. Features

### Single-Image Emotion Prediction

The `predict_image.py` script and Jupyter Notebook support:

- Accepting an image path.
- Detecting a face in the image.
- Resizing the detected face for model input.
- Predicting one of seven emotion categories.
- Displaying the predicted category and model output scores.

### Real-Time Webcam Emotion Prediction

The `real_time_emotion.py` script supports:

- Opening the webcam.
- Detecting faces in video frames.
- Predicting an emotion for each detected face.
- Displaying the predicted category on the webcam feed.

Press `q` to close the webcam window.

## 7. Installation

### Prerequisites

- Python installed on your computer.
- Git installed if cloning the repository.
- A webcam for real-time prediction.

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/ayshasana786/Face-Emotion-Recognition.git
   ```

2. Navigate to the project folder:

   ```bash
   cd Face-Emotion-Recognition
   ```

3. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

4. Activate the virtual environment on Windows:

   ```powershell
   venv\Scripts\activate
   ```

5. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

6. Ensure that the trained model is available at:

   ```text
   models/Final_model_95p07.h5
   ```

   The dataset is required only if you want to explore the data or reproduce the training process.

## 8. Running the Application

### A. Predict Emotion from an Image

Run:

```bash
python predict_image.py
```

When prompted, enter the path to an image containing a face.

Example:

```text
Enter the image path: test_images/sample_face.jpg
```

You can enter a relative or absolute path. Make sure the image exists at the specified location.

To use the Jupyter Notebook instead, open `face_emotion_recognition.ipynb` in VS Code and run the model-loading, emotion-label, and single-image prediction cells in order.

### B. Real-Time Webcam Prediction

Run:

```bash
python real_time_emotion.py
```

The application opens the webcam and displays the predicted emotion for detected faces.

Press `q` to exit.

## 9. Model and Evaluation

The trained model is stored in `models/Final_model_95p07.h5`.

The model predicts seven emotion categories. The filename does not, by itself, establish the model's accuracy.

A separate evaluation using held-out test data and appropriate classification metrics is needed to report reliable performance.

## 10. Limitations

- Predictions may be affected by lighting, face position, camera quality, and image conditions.
- Some facial expressions can be difficult to distinguish.
- Differences in the number of training examples across emotion classes may affect prediction quality.
- The model predicts facial-expression categories; it cannot determine a person's actual emotional state with certainty.

## 11. Future Improvements

- Evaluate performance on a separate test dataset.
- Report precision, recall, F1-score, and a confusion matrix.
- Improve recognition of underrepresented emotion classes.
- Test performance under different lighting and camera conditions.
- Explore improved face detection and preprocessing methods.

## Disclaimer

This project is intended for educational purposes and demonstrates deep learning and computer vision techniques. It is not intended for medical, psychological, employment, or other high-stakes decisions.
