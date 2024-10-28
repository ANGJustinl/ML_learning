from ultralytics import YOLO

# Load a model
model = YOLO("intro_test\Yolo_v10\models\yolov10s.pt")
# Export the model
model.export(format="onnx", dynamic=True)
