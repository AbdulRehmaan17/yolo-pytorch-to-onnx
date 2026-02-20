from ultralytics import YOLO
import cv2

# Load the trained YOLO model
model = YOLO("yolo11n.pt")

# Run inference on image
results = model("image.png")

# Get plotted image with detections
annotated_frame = results[0].plot()

# Save the output image
cv2.imwrite("output.png", annotated_frame)

print("Inference completed. Output saved as output.png")
