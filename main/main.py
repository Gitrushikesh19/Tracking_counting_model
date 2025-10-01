import cv2
from yolo_detection import detect_objects
from model.multi_object_tracking import MultiObjectTracker

def main():
    mot = MultiObjectTracker()

    cap = cv2.VideoCapture("../VehicleTraffic.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = detect_objects(frame)

        tracked_objects, count = mot.update_prev(detections)

        for tid, box in tracked_objects.items():
            if box is None:
                continue
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
            cv2.putText(frame, f"ID {tid}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 2)

        for (x1, y1, x2, y2) in detections:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        cv2.line(frame, (0, mot.line_position), (frame.shape[1], mot.line_position), (0, 0, 255), 2)

        cv2.putText(frame, f"Count: {count}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(0) & 0xff == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
