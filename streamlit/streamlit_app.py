import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from streamlit_cam import EmotionDetector
from torchvision import transforms
from streamlit_webrtc import webrtc_streamer
from inference.inference import Model
import streamlit as st

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Model().to(device)
model.load_state_dict(torch.load('model.pth', map_location=device, weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

st.set_page_config(layout="centered")
st.title("Real-Time Emotion Recognition")
st.write("Uses your webcam to detect facial emotion.")

def detector_factory() -> EmotionDetector:
    return EmotionDetector(model, transform, device)

webrtc_streamer(key="emotion-detection", video_processor_factory=detector_factory)
