# 24.9.21 使用torch 支持cuda
import cv2
import time
import torch
from logger import logger
from ultralytics import YOLO

from utils import frame_rate_caculate

model = YOLO("Vision\models\yolov10n.pt")
logger.info("Pre-trained YOLOv10s Model loaded")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Device using: {device}")


def Predict(model, img, classes=[], min_conf=0.5, device="cpu"):
    """
    Using Predict Model to predict objects in img.

    Input classes to choose which to output.

    eg. Predict(chosen_model, img_input, classes=[human], min_conf=0.5)
    """
    if classes:
        results = model.predict(
            img, classes=classes, conf=min_conf, device=device, stream=True
        )
    else:
        results = model.predict(img, conf=min_conf, device=device, stream=True)
    return results


def Predict_and_detect(
    model,
    img,
    classes=[],
    min_conf=0.5,
    rectangle_thickness=2,
    text_thickness=1,
    device="cpu",
):
    """
    Using Predict Model to predict objects in img and detect the objects out.

    Input classes to choose which to output.

    eg. Predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1)
    """
    results = Predict(model, img, classes, min_conf=min_conf, device=device)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(
                img,
                (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                (int(box.xyxy[0][2]), int(box.xyxy[0][3])),
                (255, 0, 0),
                rectangle_thickness,
            )
            cv2.putText(
                img,
                f"{result.names[int(box.cls[0])]}",
                (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 0, 0),
                text_thickness,
            )
    return img, results


camera = cv2.VideoCapture(0)
while True:
    # read frame
    ret, frame = camera.read()
    start_time = time.time()  # 计算帧数

    # Perform object detection on an image
    result_img, _ = Predict_and_detect(
        model, frame, classes=[], min_conf=0.5, device=device
    )
    end_time = time.time()
    result_img = frame_rate_caculate(result_img, start_time, end_time)

    # Display results
    cv2.imshow("YOLOv10 Inference", result_img)
    key = cv2.waitKey(1)
    if key == 32:  # 空格
        break

camera.release()
cv2.destroyAllWindows()
