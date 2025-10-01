from ultralytics import YOLO

yolo = YOLO("../yolov8n.pt")
vehicle_classes = [2, 3, 5, 7]

def detect_objects(frame):
    model = yolo.predict(frame, imgsz=640, conf=0.6, verbose=False)[0]
    detections = []
    for box in model.boxes:
        cls = int(box.cls[0].item())
        if cls in vehicle_classes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            detections.append([int(x1), int(y1), int(x2), int(y2)])

    return detections
