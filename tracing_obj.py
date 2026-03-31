import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO('E:/programming/bdd100k/runs/detect/train6/weights/best.onnx')

# Список цветов для различных классов
colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (192, 192, 192), (128, 128, 128), (128, 0, 0), (128, 128, 0),
    (0, 128, 0), (128, 0, 128), (0, 128, 128), (0, 0, 128), (72, 61, 139),
    (47, 79, 79), (47, 79, 47), (0, 206, 209), (148, 0, 211), (255, 20, 147)
]
input_video = 'test files/test_video1.mp4'

capture = cv2.VideoCapture(input_video)

fps = int(capture.get(cv2.CAP_PROP_FPS))
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

output_video = 'detect.mp4'

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))


while True:
    ret, frame = capture.read()

    if not ret:
        break

    res = model(frame)[0]
    

    classes_names = res.names
    classes = res.boxes.cls.cpu().numpy()
    boxes = res.boxes.xyxy.cpu().numpy().astype(np.int32)

    #рисование рамок и подписей в кадре
    for class_id, box, conf in zip(classes, boxes, res.boxes.conf):
        if conf>0.45:
            class_name = classes_names[int(class_id)]
            color = colors[int(class_id) % len(colors)]
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Запись обработанного кадра в выходной файл
    writer.write(frame)

# Освобождение ресурсов и закрытие окон
capture.release()
writer.release()

