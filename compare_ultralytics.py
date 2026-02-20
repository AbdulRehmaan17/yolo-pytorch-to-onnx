from ultralytics import YOLO

# ----------------------
# Load PyTorch Model
# ----------------------
pt_model = YOLO("yolo11n.pt")
pt_results = pt_model("image.png")

# ----------------------
# Load ONNX Model
# ----------------------
onnx_model = YOLO("yolo11n.onnx")
onnx_results = onnx_model("image.png")

# ----------------------
# Compare Results
# ----------------------
print("PyTorch detections:", len(pt_results[0].boxes))
print("ONNX detections:", len(onnx_results[0].boxes))
