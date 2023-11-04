# Violence Detection System using YOLO

## Project Objective
The aim of this project is to create a real-time violence detection system using a custom-trained YOLO (You Only Look Once) model. This system is designed to analyze video feeds, identify potential violent behavior, and issue alerts for immediate action.

### Preview
![output gif](https://github.com/ishinomaki-hackathon/trained_yolov8/blob/755e475415a8dc857b9121e3c4d51aa13941bb93/results.gif)

## Methodology

### Data Preparation
A carefully curated dataset was used to train the model. The dataset includes labeled instances of 'Violent' and 'Non-Violent' behavior and is prepared to provide balanced and diverse training material.

### Model Training
We utilized a variant of the YOLO model optimized for our specific use case, focusing on real-time detection capabilities while maintaining high accuracy.

### Implementation
The `main.py` script serves as the entry point for the system, performing real-time video processing and detection using the trained YOLO model.

## System Workflow

### Step 1: General Object Detection
```python
from ultralytics import YOLO
# Load the standard YOLO model
standard_model = YOLO('path_to_standard_weights.pt')
```

## Key Components

### Email Alert System
```python
def send_frame_as_email(frame, to_email):
    # Function to send an email alert with an image frame upon violence detection
```
The system is equipped with an email notification feature that sends out an alert with the relevant frame attached when violence is detected in the video feed.<br>
**Snippets**<br><br>
![email snapshot](https://github.com/ishinomaki-hackathon/trained_yolov8/blob/755e475415a8dc857b9121e3c4d51aa13941bb93/Screenshot%202023-11-04%20at%2013.32.43.png)

**The Frame sent:** <br><br>
![Frame sent in email](https://github.com/ishinomaki-hackathon/trained_yolov8/blob/755e475415a8dc857b9121e3c4d51aa13941bb93/frame.jpg)

## Introduction(Dual-Model Violence Detection System
This system employs two distinct YOLO models to accurately detect violence in video streams. The first model is a general YOLO detector that identifies potential subjects in the frame. When two or more individuals are detected, the second specialized pre-trained YOLO model analyzes their interactions to determine the presence of violence.

## System Workflow

### Step 1: General Object Detection
```python
from ultralytics import YOLO
# Load the standard YOLO model
standard_model = YOLO('path_to_standard_weights.pt')
The standard YOLO model is responsible for detecting the presence of people in the video feed.
```

### Step 2: Violence Detection Analysis
```python
# Load the specialized YOLO model for violence detection
violence_model = YOLO('path_to_violence_weights.pt')
```
When the standard model detects two or more people, the specialized violence detection model is engaged to further analyze the interaction for violent behavior.

### Step 3: Continuous Monitoring and Alerting
The system continuously monitors the video feed and, upon confirmation of violence lasting more than 10 frames, triggers an email alert with a snapshot of the incident.

```python
# Pseudocode for the monitoring loop
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Initial detection of individuals
    detections = standard_model.predict(frame)
    # If two or more people detected, use the violence detection model
    if len(detections) >= 2:
        violence_detections = violence_model.predict(frame)
        # If violence is detected consistently, send an email alert
        if is_violence_detected(violence_detections):
            # Send an email after 10 consistent detections
            if violence_duration > 10:
                send_frame_as_email(frame, to_email)
```
**Prompt Output Snapshot**<br><br>
![prompt output snapshot](https://github.com/ishinomaki-hackathon/trained_yolov8/blob/755e475415a8dc857b9121e3c4d51aa13941bb93/Screenshot%202023-11-04%20at%2012.58.12.png)


### Real-Time Video Processing
```python
cap = cv2.VideoCapture(0)
The script captures live video feeds, applies the trained YOLO model to detect violence, and triggers alerts when necessary.
```

## Results and Outcomes

### Dataset Correlogram
![Dataset Correlogram](https://github.com/ishinomaki-hackathon/trained_yolov8/blob/4f80aeafcec674f8fb1a30910a9af469364a9b3c/labels_correlogram.jpg)
The correlogram illustrates the correlation and distribution of different classes within our dataset, highlighting the dataset's balance and diversity.

### Confusion Matrix
![Normalized Confusion Matrix](https://github.com/ishinomaki-hackathon/trained_yolov8/blob/4f80aeafcec674f8fb1a30910a9af469364a9b3c/confusion_matrix_normalized.png)
The normalized confusion matrix displays the model's classification accuracy, demonstrating a high true positive rate and low false positive rate for the detection of violent behavior.

### Training Results
![Training Results](https://github.com/ishinomaki-hackathon/trained_yolov8/blob/4f80aeafcec674f8fb1a30910a9af469364a9b3c/results.png)
Graphs depicting the training performance of the model, illustrating the optimization of the loss function and improvements in key metrics such as precision and recall.

### Validation Predictions
![Validation preds](https://github.com/ishinomaki-hackathon/trained_yolov8/blob/4f80aeafcec674f8fb1a30910a9af469364a9b3c/val_batch2_pred.jpg)
**Predictions**<br>

![Validation labels](https://github.com/ishinomaki-hackathon/trained_yolov8/blob/4f80aeafcec674f8fb1a30910a9af469364a9b3c/val_batch2_labels.jpg)
**Ground Truth or Labels(This was still early in training).**<br>

Validation images with predictions versus actual labels provide a visual confirmation of the model's predictive accuracy on unseen data.

## Applications

The system can be utilized in a range of environments, including public spaces, schools, and by law enforcement agencies to monitor and respond to incidents of violence effectively.

## Team Collaboration

The development of this system was a collaborative effort, with team members contributing across different aspects such as dataset annotation, model training, and system integration. we developed our own dataset and decided to train it as well, but a lot of help was taken from Roboflow dataset which saved us time in annotation.

## Usage

To run the violence detection system:

- Update the data.yaml file with the correct paths to your dataset.
- Configure the email settings in main.py to receive alerts.
- Execute the [main.py](https://github.com/ishinomaki-hackathon/trained_yolov8/blob/ceab733f53d153e0e2fc24f01602d3d4243a78d9/main.py) script to start the detection system.
- There were two models used: one is for object detection which is the regular yolo and one with the custom dataset, here are the weight, you can download these [weights](https://github.com/ishinomaki-hackathon/trained_yolov8/blob/755e475415a8dc857b9121e3c4d51aa13941bb93/best.pt) and load then in the [main.py](https://github.com/ishinomaki-hackathon/trained_yolov8/blob/ceab733f53d153e0e2fc24f01602d3d4243a78d9/main.py) file.

- ## Setup and Configuration

### Prerequisites
- Python 3.6 or later
- OpenCV library
- Ultralytics YOLO library
- SMTP server access for email notifications


## Conclusion

Our violence detection system represents a significant step towards enhancing public safety through automated monitoring and real-time alerts using the power of Artificial Intelligence.

For more information on setting up and using the system, please refer to the detailed comments in the main.py script and the [Jupyter notebook](https://github.com/ishinomaki-hackathon/trained_yolov8/blob/4f80aeafcec674f8fb1a30910a9af469364a9b3c/violence_detection_custom_dataset_yolo_train.ipynb) provided in the repository.

