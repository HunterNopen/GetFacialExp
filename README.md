# GetFacialExp

GetFacialExp is a deep learning project for real-time facial expression recognition using convolutional neural networks (CNNs). The system is designed to **classify human emotions** from webcam images into seven categories: Angry, Disgusted, Fearful, Happy, Neutral, Sad and Surprised.

## Overview

The project demonstrates the full pipeline for emotion recognition:
- **Dataset preparation** using the FER2013 dataset, with automated download and preprocessing.
- **Model training** with a custom CNN architecture leveraging PyTorch.
- **Real-time inference** via webcam using OpenCV, with live emotion prediction and display.

## Techniques & Libraries Used

- **PyTorch**: Model definition, training, and inference.
- **Torchvision**: Image transformations and dataset utilities.
- **OpenCV**: Webcam capture, face detection, and visualization.
- **KaggleHub**: Automated dataset download.
- **TQDM**: Training progress visualization.

## Methodology

1. **Data Preparation**:  
   - The FER2013 dataset is downloaded and organized into train/test folders.
   - Images are converted to grayscale, resized to 48x48 pixels, and normalized.

2. **Model Architecture**:  
   - A deep CNN with multiple convolutional, batch normalization, ReLU, max pooling, and dropout layers.
   - Final classification via fully connected layers.

3. **Training**:  
   - Cross-entropy loss and AdamW optimizer.
   - Training for 10 epochs with batch size 8.

4. **Inference**:  
   - Real-time webcam feed.
   - Face detection using Haar cascades.
   - Preprocessing and emotion prediction displayed on video.

## Results

- **Final Training Loss**: Reduced from ~2.0 to ~0.5
- **Validation Accuracy**: Achieved approximately 0.85 (85%)
- **Inference Speed**: <0.2 seconds per frame on a standard webcam, but implemnted debouncing to ensure smooth video feed

## Conclusion

GetFacialExp provides an end-to-end solution for facial emotion recognition, showcasing effective use of modern deep learning and computer vision libraries. The project achieves strong accuracy and demonstrates real-time performance, making it suitable for interactive applications and further research.

---------------------------------------------------