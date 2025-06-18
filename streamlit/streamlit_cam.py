import cv2, time, torch
from streamlit_webrtc import VideoTransformerBase
import av
import streamlit as st

class EmotionDetector(VideoTransformerBase):
    def __init__(self, model, transform, device):
        self.model = model
        self.transform = transform
        self.device = device
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                             3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
        self.last_pred_time = 0
        self.cooldown = 2
        self.last_prediction = "..."

    def predict_emotion(self, img):
        tensor = self.transform(img).unsqueeze(0).to(torch.float32).to(self.device)
        with torch.no_grad():
            output = self.model(tensor)
            predicted = torch.argmax(output, dim=1).item()
        return predicted

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x,y,w,h) in faces:
            roi = gray[y:y+h, x:x+w]
            if roi.size == 0:
                continue
            resized = cv2.resize(roi, (48, 48))
            if time.time() - self.last_pred_time > self.cooldown:
                cls = self.predict_emotion(resized)
                self.last_prediction = self.emotion_dict.get(cls, "...")
                self.last_pred_time = time.time()

            cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)
            cv2.putText(image, self.last_prediction, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

        return av.VideoFrame.from_ndarray(image, format="bgr24")


singleton_detector = None

@st.cache_resource
def init_detector(model, transform, device):
    global singleton_detector
    if singleton_detector is None:
        singleton_detector = EmotionDetector(model, transform, device)
    return singleton_detector