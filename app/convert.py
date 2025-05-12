import os
import subprocess

# 🔹 Конвертація ONNX моделі в OpenVINO формат
# subprocess.run([
#     "mo",
#     "--input_model", "models/yolo11n-old.onnx",
#     "--output_dir", "openvino_model",
#     "--model_name", "yolo11n"
# ], check=True)

# 🔹 Конвертація ONNX моделі в NCNN формат
from ultralytics import YOLO
# Load a YOLO11n PyTorch model
model = YOLO("./models/yolo11n.pt")

model.export(format='ncnn', half=True, batch=1, device="cpu")
# Export the model to NCNN format
# model.export(format="onnx")  # creates 'yolo11n_ncnn_model'

# Load the exported NCNN model
# ncnn_model = YOLO("yolo11n_ncnn_model")
