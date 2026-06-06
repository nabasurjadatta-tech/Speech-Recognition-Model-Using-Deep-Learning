# Speech-Recognition-Model-Using-Deep-Learning
Designed and Trained a deep learning model in Medical diagnosis to classify respiratory health conditions like Asthma, Pneumonia, Bronchitis, COPD and healthy from patient speech and breathing audio recordings. Extracted MFCC features, applied audio augmentation and trained a C NN model . Achieved a test accuracy of 86.48%. 

Project Workflow / Steps Involved

1. Problem Definition
Develop an AI model that can detect asthma conditions from respiratory audio signals.
Classify audio recordings into the appropriate respiratory health category.

2. Import Required Libraries

Libraries used:

NumPy
Librosa
TensorFlow/Keras
Matplotlib
Seaborn
Scikit-learn

Purpose:

Audio processing
Feature extraction
Deep learning model building
Evaluation and visualization

3. Load Dataset
https://drive.google.com/drive/folders/1x_9HXbhdLNR-MNilA81Hu7QIZdO2QSt1
Audio files are organized into folders based on class labels.
Each folder represents a category.

5. Audio Visualization

Generate:

Spectrograms
MFCC visualizations

Purpose:

Understand audio characteristics.
Verify dataset quality.
4. Audio Data Augmentation

Techniques applied:

Noise addition
Time stretching
Pitch shifting

Purpose:

Increase dataset size.
Improve model generalization.
Reduce overfitting.

5. Feature Extraction using MFCC

Extract:
40 MFCC coefficients
Process:
Load audio file.
Compute MFCC features.
Pad/trim features to fixed length (100 frames)
Output Shape:
40 × 100

6. Dataset Preparation

Function:

load_audio_data()

Tasks:

Read all audio files.
Extract MFCC features.
Store features and labels.

7. Label Encoding

Convert class names into numerical labels.

Example:

Healthy → 0
Asthma → 1
Other → 2

8. Train-Test Split
80% Training
20% Testing

Purpose:

Train on one portion.
Evaluate on unseen data.

9. Data Reshaping for CNN

CNN expects 4D input:

(samples, 40, 100, 1)

where:

40 = MFCC coefficients
100 = Time frames
1 = Channel
11. One-Hot Encoding

Convert labels into categorical format:

0 → [1,0,0]
1 → [0,1,0]
2 → [0,0,1]

10. Handle Class Imbalance

Compute:

class_weights

Purpose:

Prevent bias toward majority classes.
Improve fairness of training.

11. Build CNN Model

Architecture:

Input Layer
↓
Conv2D (32 filters)
↓
MaxPooling
↓
Dropout
↓
Conv2D (64 filters)
↓
MaxPooling
↓
Dropout
↓
Dense Layers
↓
Softmax Output Layer

Purpose:

Learn patterns from MFCC feature maps.

12. Compile Model

Configuration:

Optimizer = Adam
Loss = Categorical Crossentropy
Metric = Accuracy
15. Configure Training Callbacks
Model Checkpoint
best_model.keras

Saves the best model.

Early Stopping

Stops training if validation performance stops improving.

13. Train the CNN
model.fit()

Parameters:

Epochs = 30
Batch Size = 32

Outputs:

Training Accuracy
Validation Accuracy
Training Loss
Validation Loss

14. Load Best Model
load_model("best_model.keras")

Purpose:

Use the best-performing version for evaluation.

15. Model Evaluation

Generate predictions on test data:

y_pred = model.predict(X_test)

16. Training Accuracy

Evaluate performance on training set.

Output:

Train Accuracy = xx.xx%

17. ROC Curve Analysis

Compute:

False Positive Rate (FPR)
True Positive Rate (TPR)
AUC Score

Purpose:

Measure classification quality.

18. Classification Report

Metrics:

Precision
Recall
F1 Score
Support

Provides detailed class-wise performance.

19. Confusion Matrix

Visualize:

Correct predictions
Misclassifications

Helps understand model errors.

20. Accuracy & Loss Graphs

Plot:

Training Accuracy
Validation Accuracy
Training Loss
Validation Loss

Purpose:

Detect overfitting or underfitting.

21. Real-Time Prediction Function
predict_audio(file_path)

Steps:

Load new audio.
Extract MFCC.
Reshape input.
Predict using CNN.
Return class label.

22. Testing on New Audio Files

Example:

predict_audio("new_audio.wav")

Output:

Predicted Class: Asthma

23. Final Testing

Evaluate on test set:

Test Accuracy = xx.xx%
Architecture Summary
Audio Dataset
       ↓
Preprocessing
       ↓
Data Augmentation
       ↓
MFCC Feature Extraction
       ↓
Train-Test Split
       ↓
CNN Model
       ↓
Training
       ↓
Evaluation
       ↓
ROC + Confusion Matrix
       ↓
Real-Time Asthma Prediction

