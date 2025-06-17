import cv2
import time
import torch
from streamlit_webrtc import VideoTransformerBase
import av
import streamlit as st

class EmotionDetector(VideoTransformerBase):
    def __init__(self, model, transform, device):
        self.model = model
        self.device = device

        self.model.to(self.device)
        self.model.eval()

        self.transform = transform

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.emotion_dict = {
            0: "Angry", 1: "Disgusted", 2: "Fearful",
            3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"
        }

        self.last_pred_time = 0
        self.cooldown = 2
        self.last_prediction = "..."

        self.debug_area = st.sidebar.empty()

    def predict_emotion(self, img):
        tensor = self.transform(img).unsqueeze(0).to(torch.float32).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
            predicted_class = torch.argmax(output, dim=1).item()
        self.debug_area.text(f"Prediction: {predicted_class}")
        return predicted_class
    
    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            if roi_gray.size == 0:
                continue

            resized = cv2.resize(roi_gray, (48, 48))

            if time.time() - self.last_pred_time > self.cooldown:
                pred_class = self.predict_emotion(resized)
                self.last_prediction = self.emotion_dict.get(pred_class, "...")
                self.last_pred_time = time.time()

            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(image, self.last_prediction, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(image, "Running", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame(image)