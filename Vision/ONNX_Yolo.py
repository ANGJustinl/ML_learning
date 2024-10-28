# https://blog.csdn.net/qq_42589613/article/details/140040501
# coding:utf-8
import cv2
import onnxruntime as rt
import torch
from logger import logger
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml

from utils import *


coco_yaml_path = "Vision\coco8.yaml"
Model_path = "Vision\models\yolov10s_hg.onnx"

with open(coco_yaml_path, "r") as config:
    config = yaml_load(check_yaml(coco_yaml_path))
std_h, std_w = 640, 640  # 标准输入尺寸
dic = config["names"]  # 得到的是模型类别字典
class_list = list(dic.values())

if torch.cuda.is_available():
    logger.info("Using CUDA")
    providers = ["CUDAExecutionProvider"]
else:
    logger.info("Using CPU")
    providers = ["CPUExecutionProvider"]


sess = rt.InferenceSession(
    "intro_test\Yolo_v10\models\yolov10m_hg.onnx", providers=providers
)  # yolov10模型onnx格式
logger.info("Pre-trained YOLOv10s ONNX Model loaded")


camera = cv2.VideoCapture(0)
while True:
    # read frame
    ret, frame = camera.read()
    start_time = time.time()  # 计算帧数
    # 前处理
    img_after = resize_image(frame, (std_w, std_h), True)  # （640， 640， 3）
    # 将图像处理成输入的格式
    data = img2input(img_after)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred = sess.run([label_name], {input_name: data})[
        0
    ]  # 输出(8400x84, 84=80cls+4reg, 8400=3种尺度的特征图叠加), 这里的预测框的回归参数是xywh， 而不是中心点到框边界的距离
    pred = std_output(pred)
    # 置信度过滤+nms
    result = nms(pred, 0.7, 0.4)  # [x,y,w,h,conf(最大类别概率),class]
    # 坐标变换
    result = cod_trf(result, frame, img_after)
    image = draw(result, frame, class_list)
    end_time = time.time()
    image = frame_rate_caculate(image, start_time, end_time)
    # Display results
    cv2.imshow("YOLOv10 Inference", image)
    key = cv2.waitKey(1)
    if key == 32:  # 空格
        break

camera.release()
cv2.destroyAllWindows()
