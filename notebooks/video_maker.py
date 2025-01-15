import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from IPython.display import Video
from IPython.display import HTML
from IPython.display import display




filename = '/home/iulian/Downloads/Simple_Cholecystectomy.mp4'
output_folder = '/home/iulian/chole_ws/src/sam2/Simple_Cholecystectomy_frames'
cap = cv2.VideoCapture(filename)
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
frames = []
## read frames from start_idx to end_idx
start_idx = 1120
end_idx = 250 + start_idx
cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
    if len(frames) == end_idx - start_idx:
        break
cap.release()
print(len(frames))

## save frames to output folder
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
for i, frame in enumerate(frames):
    cv2.imwrite(os.path.join(output_folder, f'{i}.png'), frame)


