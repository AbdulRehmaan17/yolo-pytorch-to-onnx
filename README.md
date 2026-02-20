YOLO PyTorch to ONNX Conversion

This project demonstrates the complete workflow of converting a YOLO11n object detection model from PyTorch format (.pt) to ONNX format (.onnx), and validating inference consistency between both models.

🚀 Project Workflow

Load YOLO11n PyTorch model using Ultralytics
Perform object detection inference on a sample image
Export the model to ONNX format
Run inference using ONNX Runtime
Compare detection outputs between PyTorch and ONNX models
Validate detection consistency and performance

🛠 Technologies Used

Python
Ultralytics YOLO
PyTorch
ONNX
ONNX Runtime
OpenCV
NumPy

✅ Key Outcome

Both PyTorch and ONNX models produced identical detection results, confirming successful model export and inference compatibility.
