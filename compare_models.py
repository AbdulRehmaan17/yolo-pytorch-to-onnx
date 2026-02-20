from ultralytics import YOLO
import onnxruntime as ort
import numpy as np
import cv2

# ----------------------
# Load Image (PNG)
# ----------------------
image_path = "image.png"

img = cv2.imread(image_path)

if img is None:
    print("Error: Image not found. Make sure image.png is in the project folder.")
    exit()

img_resized = cv2.resize(img, (640, 640))
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
img_normalized = img_rgb / 255.0

input_tensor = np.transpose(img_normalized, (2, 0, 1))
input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)

# ----------------------
# PyTorch Inference
# ----------------------
model = YOLO("yolo11n.pt")
results = model(image_path)

if len(results[0].boxes) == 0:
    print("No objects detected in PyTorch model.")
    exit()

pt_boxes = results[0].boxes.xyxy.cpu().numpy()
pt_box = pt_boxes[0]  # take first detected box

# ----------------------
# ONNX Inference
# ----------------------
session = ort.InferenceSession("yolo11n.onnx")
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: input_tensor})

onnx_output = outputs[0][0]  # shape (84, 8400)

# Extract highest confidence detection
scores = onnx_output[4:, :]
conf_scores = scores.max(axis=0)
max_conf_index = np.argmax(conf_scores)

onnx_box = onnx_output[:4, max_conf_index]

# ----------------------
# IoU Calculation
# ----------------------
def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = max(0, (box1[2] - box1[0])) * max(0, (box1[3] - box1[1]))
    box2_area = max(0, (box2[2] - box2[0])) * max(0, (box2[3] - box2[1]))

    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0

    return inter_area / union_area

iou = calculate_iou(pt_box, onnx_box)

print("PyTorch Box:", pt_box)
print("ONNX Box:", onnx_box)
print("IoU between PyTorch and ONNX:", iou)
