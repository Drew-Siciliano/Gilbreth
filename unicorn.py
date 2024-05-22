from ultralytics import YOLO
import torch

model = YOLO("runs/detect/train38/weights/last.pt")
#model = YOLO('yolov8x.pt')

results = model.train(data = "/scratch/gilbreth/gsicilia/UNICORN/data.yaml", epochs = 100, imgsz = 800, batch = 15, workers = 1)
# results = model.train(data = "/scratch/gilbreth/gsicilia/xView/dataYAML.yaml", epochs = 100, imgsz = 800, amp=False)
