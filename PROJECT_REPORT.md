# Hand Gesture Recognition Model Using Deep Learning

## Project Report

**Project Title:** Hand Gesture Recognition Model  
**Author:** Sarat Banik  
**GitHub Repository:** [Hand-Gesture-Recognition-Model](https://github.com/saratbanik/Hand-Gesture-Recognition-Model)  
**Dataset:** [LEAPGestRecog Dataset - Kaggle](https://www.kaggle.com/datasets/gti-upm/leapgestrecog)  
**Programming Language:** Python  
**Model:** Convolutional Neural Network (CNN)

---

## 1. Abstract

Hand gesture recognition is an important computer vision application that enables people to interact with computers through natural hand movements. This project develops a deep-learning-based system to classify hand gesture images into ten categories using a Convolutional Neural Network (CNN).

The system uses the LEAPGestRecog dataset and includes image preprocessing, dataset splitting, model training, evaluation, and prediction. It also provides a Streamlit web application for image-based predictions and an optional OpenCV webcam demonstration.

The project demonstrates a practical machine learning workflow, from preparing the dataset to using a trained model for gesture classification. Actual performance is evaluated after training using the generated evaluation outputs.

## 2. Introduction

Human-computer interaction commonly relies on keyboards, mice, and touchscreens. Hand gesture recognition provides an alternative approach that can support touch-free interaction and accessibility-oriented applications.

Deep learning enables a computer to learn visual patterns from labelled images and use those patterns to classify new inputs. Convolutional Neural Networks are particularly useful for image classification because they can learn spatial features from image data.

This project applies a CNN to recognize ten different hand gestures. It combines image preprocessing, model development, evaluation, and user interfaces to demonstrate how computer vision can be used for gesture-based interaction.

## 3. Problem Statement

Manually identifying hand gestures is unsuitable for applications that require automatic and repeated recognition. A computerized solution is needed to analyze an input image and predict its corresponding hand gesture.

The main challenge is to develop a model that learns meaningful visual features from labelled images and can classify gestures from data that was not used during training.

## 4. Objectives

The primary objectives of this project are:

1. Develop a CNN-based hand gesture classification model.
2. Recognize ten different hand gesture categories.
3. Preprocess images into a consistent format for model training and prediction.
4. Split the dataset by subject to reduce data leakage between related images.
5. Evaluate model performance using appropriate evaluation metrics.
6. Provide a command-line tool for image prediction.
7. Develop a Streamlit interface for interactive predictions.
8. Provide an optional webcam demonstration for experimentation.

## 5. Scope of the Project

The project focuses on image-based hand gesture classification using the LEAPGestRecog dataset.

It covers dataset organization, image preprocessing, model training, model evaluation, saving the trained model, and predicting gestures from new images.

The optional webcam component demonstrates a possible real-time use case. However, webcam predictions may be unreliable because the dataset's image characteristics can differ from images captured by an ordinary RGB webcam.

This project is intended for educational and experimental purposes.

## 6. Dataset Description

The project uses the **LEAPGestRecog dataset**, available on Kaggle.

**Dataset link:** https://www.kaggle.com/datasets/gti-upm/leapgestrecog

The dataset contains images organized by subjects and gesture categories. These labelled images are used to train and evaluate the classification model.

### 6.1 Gesture Categories

The model is designed to classify the following ten categories:

| Dataset Label | Gesture |
|---|---|
| `01_palm` | Palm |
| `02_l` | L |
| `03_fist` | Fist |
| `04_fist_moved` | Moving Fist |
| `05_thumb` | Thumb |
| `06_index` | Index |
| `07_ok` | OK |
| `08_palm_moved` | Moving Palm |
| `09_c` | C |
| `10_down` | Down |

The dataset must be downloaded and extracted locally before training. Its large image collection should not be uploaded to GitHub as part of the source repository.

## 7. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow and Keras | Building, training, and loading the CNN model |
| NumPy | Numerical operations and array processing |
| Pillow | Image loading and conversion |
| scikit-learn | Model evaluation and classification metrics |
| Matplotlib | Generating training and evaluation visualizations |
| Streamlit | Building the interactive web interface |
| OpenCV | Capturing and displaying webcam video |

## 8. Project Structure

The project is organized into the following main files:

```text
Hand-Gesture-Recognition-Model/
│
├── app.py
├── train_model.py
├── dataset_utils.py
├── predict_image.py
├── webcam_demo.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
│
├── data/
│   └── leapGestRecog/
│       └── leapGestRecog/
│           ├── 00/
│           ├── 01/
│           ├── ...
│           └── 09/
│
└── artifacts/
    ├── hand_gesture_model.keras
    ├── best_validation_model.keras
    ├── class_names.json
    ├── metrics.json
    ├── confusion_matrix.png
    └── training_history.png
```

The folder nesting may vary depending on how the dataset is extracted. The `artifacts/` directory and its files are generated during training.

### 8.1 Description of Important Files

- **`dataset_utils.py`** — Discovers dataset images, collects labels and subject identifiers, splits the dataset by subject, and preprocesses images.
- **`train_model.py`** — Trains the CNN, evaluates the model, and saves the model and evaluation outputs.
- **`predict_image.py`** — Predicts the gesture shown in a supplied image.
- **`app.py`** — Provides an interactive Streamlit application for image-based prediction.
- **`webcam_demo.py`** — Provides an optional webcam demonstration using OpenCV.
- **`requirements.txt`** — Lists the Python dependencies required by the project.
- **`README.md`** — Contains setup instructions and usage information.
- **`PROJECT_REPORT.md`** — Documents the project's objectives, methodology, implementation, and evaluation.

## 9. System Architecture

The system follows a sequence of data preparation, training, evaluation, and inference stages.

```text
        LEAPGestRecog Dataset
                  |
                  v
       Dataset Discovery and
          Image Preprocessing
                  |
                  v
       Subject-Based Data Split
                  |
                  v
        CNN Model Training
                  |
          +-------+-------+
          |               |
          v               v
     Saved Model     Model Evaluation
          |           and Metrics
          v
      Prediction Interfaces
          |
    +-----+----------+
    |        |       |
    v        v       v
  Image   Streamlit  Webcam
Prediction   App      Demo
```

The trained model is saved and reused by the prediction interfaces, avoiding the need to train the model every time a prediction is made.

## 10. Methodology

### 10.1 Dataset Preparation

The dataset is downloaded from Kaggle and extracted into the project's `data/` directory. The utility script identifies image files and associates each image with its gesture label and subject identifier.

### 10.2 Image Preprocessing

Before training and prediction, images are processed consistently. The planned preprocessing pipeline includes:

1. Loading the image.
2. Converting it to grayscale.
3. Resizing it to 96 × 96 pixels.
4. Normalizing pixel values for neural-network input.
5. Preparing the image in the format expected by the model.

Consistent preprocessing helps ensure that training images and prediction images are represented in the same format.

### 10.3 Dataset Splitting

The implementation uses subject-based data splitting to reduce the chance that similar images from the same subject appear in both training and testing data.

For the expected subject folders `00` through `09`, the intended split is:

| Dataset Portion | Subject Folders | Purpose |
|---|---|---|
| Training | `00`–`07` | Train the model |
| Validation | `08` | Monitor performance during training |
| Testing | `09` | Evaluate performance on a held-out subject |

The actual split should be verified against the extracted dataset structure before interpreting evaluation results.

### 10.4 CNN Model Training

A Convolutional Neural Network learns image features from the training dataset and uses them to distinguish the ten gesture categories.

The model is trained on labelled images. The validation data helps monitor model performance during development. Once training and evaluation are complete, the model and supporting files are saved for inference.

### 10.5 Gesture Prediction

For a new image, the system applies the same preprocessing pipeline used during training. The processed image is passed to the trained model, which produces scores for the supported gesture classes.

The class with the highest predicted score is selected as the predicted gesture. The application can also display the associated confidence score.

### 10.6 Model Evaluation

The evaluation process uses the held-out test data to assess how well the model classifies previously unseen subjects.

The project generates evaluation artifacts, including metrics and plots. The confusion matrix can reveal which gestures the model confuses, while the training history can help identify possible overfitting.

Actual evaluation values must be taken from the outputs of the training run.

## 11. Implementation and Execution

### 11.1 Environment Setup

Install a compatible Python version and create a virtual environment:

```bash
python -m venv venv
```

Activate it in Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install the project dependencies:

```bash
python -m pip install -r requirements.txt
```

Python 3.10–3.12 is recommended for the TensorFlow version specified in the project's dependency file.

### 11.2 Train the Model

After downloading and extracting the dataset, run:

```bash
python train_model.py
```

The script should save the trained model, class names, evaluation metrics, and visualization files in the `artifacts/` directory.

### 11.3 Predict an Image

Use the following command to classify an image:

```bash
python predict_image.py --image "C:\path\to\your\gesture.png"
```

Replace the example path with the actual path to your image.

### 11.4 Run the Web Application

Start the Streamlit interface:

```bash
streamlit run app.py
```

Open the local URL shown in the terminal and upload a supported image to view the model's prediction.

### 11.5 Run the Optional Webcam Demo

Start the webcam demonstration:

```bash
python webcam_demo.py
```

Press **Q** in the webcam window to exit.

Webcam predictions are experimental because the camera input may differ from the dataset used for training.

## 12. Results and Discussion

The results of the project must be recorded after executing the training and evaluation scripts. Numerical values should not be reported until they have been generated and verified.

| Evaluation Measure | Result |
|---|---|
| Test accuracy | To be filled after training |
| Precision | To be filled if calculated by the evaluation script |
| Recall | To be filled if calculated by the evaluation script |
| F1-score | To be filled if calculated by the evaluation script |
| Best validation performance | To be filled from the generated metrics |
| Test subject | Confirm from the actual data split; expected subject `09` |

The confusion matrix should be examined to identify correctly classified gestures and frequently confused classes. Training and validation plots should also be compared to assess whether the model generalizes beyond the training data.

Any final conclusion about model accuracy should be supported by the actual evaluation results.

## 13. Limitations

1. **Dataset dependency:** The model's performance depends on the quality and characteristics of the training data.
2. **Domain differences:** Images from a normal RGB webcam may differ from LEAPGestRecog images, reducing prediction accuracy.
3. **Environmental variation:** Lighting, orientation, background, and camera characteristics can affect predictions.
4. **Generalization:** Testing on a held-out subject provides useful evidence but does not guarantee performance in every real-world setting.
5. **Computational requirements:** Training time depends on dataset size and available hardware.
6. **Experimental status:** The current system is an educational prototype and should not be used for safety-critical controls without additional validation.

## 14. Future Enhancements

Potential improvements include:

- Testing the model on more diverse images captured by ordinary RGB cameras.
- Adding hand detection or segmentation before classification.
- Experimenting with data augmentation and alternative CNN architectures.
- Improving the interface with clearer error messages and additional prediction visualizations.
- Optimizing inference for low-latency applications.
- Exploring integration with accessibility tools and gesture-controlled interfaces.

## 15. Applications

Potential application areas include:

- Touch-free human-computer interaction
- Gesture-based interface control
- Educational computer vision demonstrations
- Accessibility-oriented interface experiments
- Experimental smart-device control

These are potential uses; the current project does not establish production-level reliability in these applications.

## 16. Conclusion

This project demonstrates an end-to-end workflow for hand gesture image classification using deep learning. It combines dataset preparation, image preprocessing, subject-based splitting, CNN training, model evaluation, and several prediction interfaces.

The project provides practical experience with TensorFlow/Keras, image processing, model evaluation, and Streamlit application development. Its final effectiveness must be assessed using the measured results from the held-out test data. Further testing with images from ordinary cameras would be necessary before drawing conclusions about real-world performance.

## 17. References

1. GTI-UPM, **LEAPGestRecog Dataset**, Kaggle.  
   https://www.kaggle.com/datasets/gti-upm/leapgestrecog

2. TensorFlow, **TensorFlow Documentation**.  
   https://www.tensorflow.org/learn

3. Streamlit, **Streamlit Documentation**.  
   https://docs.streamlit.io/

4. OpenCV, **OpenCV Documentation**.  
   https://docs.opencv.org/

5. scikit-learn, **scikit-learn Documentation**.  
   https://scikit-learn.org/stable/

## 18. Project Repository

GitHub: https://github.com/saratbanik/Hand-Gesture-Recognition-Model

---

**Final note:** Update the Results and Discussion section after running `train_model.py`. Include only metrics that are actually produced by your implementation.