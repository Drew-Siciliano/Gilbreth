from ultralytics import YOLO
import torch

model = YOLO('yolov8x.pt')

results = model.train(data = "/scratch/gilbreth/gsicilia/xView/dataYAML.yaml", epochs = 100, imgsz = 800, batch = 3)
#results = model.train(data = "/scratch/gilbreth/gsicilia/xView/dataYAML.yaml", epochs = 100, imgsz = 800, amp=False)
