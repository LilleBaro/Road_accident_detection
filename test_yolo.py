from ultralytics import YOLO
import torch
print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
model = YOLO("models/yolo12s.pt")  # ou YOLO("yolov8n.pt") si tu veux tester avec un modèle standard
print("Model loaded OK")
