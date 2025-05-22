import cv2
import torch
from torchvision import transforms
from inference import Model
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f'\n{device}')

model = Model()
model.load_state_dict(torch.load('model.pth'))

inference_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

def predict(model, img):
    model.eval()
    with torch.no_grad():
        tensor = inference_transform(img).unsqueeze(0).to(torch.float32).to(device)
        output = model(tensor)
        predicted_class = torch.argmax(output, dim=1)
    return predicted_class.item()

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

cam = cv2.VideoCapture(0)
last_pred_time = 0

while True:
    current_time = time.time()
    should_predict = current_time - last_pred_time > 2

    ret, frame = cam.read()

    if not ret:
        break

    bounding_box = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame, minSize=(200,200), minNeighbors=5)

    for (x,y,w,h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_img = gray_frame[y:y + h, x: x + w]
        cropped_img = cv2.resize(roi_gray_img, (48, 48))

        if should_predict:
            current_prediction = predict(model, cropped_img)
            last_pred_time = current_time

        cv2.putText(frame, emotion_dict[current_prediction], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Video Capture', cv2.resize(frame, (1200, 860), interpolation=cv2.INTER_CUBIC))

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        cam.release()
        cv2.destroyAllWindows()
        break