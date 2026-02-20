import onnxruntime as ort
import numpy as np
import cv2

# Load ONNX model
session = ort.InferenceSession("yolo11n.onnx")

# Load image
img = cv2.imread("image.png")
img_resized = cv2.resize(img, (640, 640))
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
img_normalized = img_rgb / 255.0

# Change shape to (1, 3, 640, 640)
input_tensor = np.transpose(img_normalized, (2, 0, 1))
input_tensor = np.expand_dims(input_tensor, axis=0).astype(np.float32)

# Get input name
input_name = session.get_inputs()[0].name

# Run inference
outputs = session.run(None, {input_name: input_tensor})

print("ONNX Model Inference Successful!")
print("Output shape:", outputs[0].shape)
