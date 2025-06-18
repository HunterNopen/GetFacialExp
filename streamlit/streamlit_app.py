from streamlit_cam import init_detector
from inference.inference import Model
import torch
from torchvision import transforms
from streamlit_webrtc import webrtc_streamer
import streamlit as st

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

st.set_page_config(layout="centered")
st.title("Real-Time Emotion Recognition")

detector_factory = lambda: init_detector(model, transform, device)

webrtc_streamerkey=webrtc_streamer("emotion-detection", video_processor_factory=lambda: init_detector())