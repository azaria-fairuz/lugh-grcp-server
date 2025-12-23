import os
import cv2
import time
import numpy as np
from ultralytics import YOLO

def roi_preprocess_image(frame):
    b, g, r = cv2.split(frame)
    imc = np.maximum(np.maximum(r, g), b)
    imc = imc.astype(np.uint8)
    imc = cv2.GaussianBlur(imc, (5,5), 0)

    atg = cv2.adaptiveThreshold(imc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    _, bw = cv2.threshold(atg, 130, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3,3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    rgb = cv2.cvtColor(bw, cv2.COLOR_BGR2RGB)
    
    return rgb

def needle_preprocess_image(frame):
    b, g, r = cv2.split(frame)
    imc = np.maximum(np.maximum(r, g), b)
    imc = imc.astype(np.uint8)

    atg = cv2.adaptiveThreshold(imc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    _, bw = cv2.threshold(atg, 130, 255, cv2.THRESH_BINARY_INV)
    rgb = cv2.cvtColor(bw, cv2.COLOR_BGR2RGB)
    
    return rgb

win_name   = "GAUGE-DETECTION"
rtsp_url   = "rtsp://admin:Pdu12345678@191.101.190.233:8554/Streaming/Channels/101?rtsp_transport=tcp"
roi_model_path = os.path.join(os.getcwd(), 'models', 'gauge-roi.pt')
needle_model_path = os.path.join(os.getcwd(), 'models', 'gauge-needle.pt')

cap = cv2.VideoCapture(rtsp_url)
roi_model = YOLO(roi_model_path)
needle_model = YOLO(needle_model_path)

cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

while True:
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print("Failed to open RTSP stream. Retrying...")
        time.sleep(5)
        continue
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lost connection. Reconnecting...")
            cap.release() 
            time.sleep(5)
            break 
    
        roi_image = roi_preprocess_image(frame)
        roi_results = roi_model.predict(source=roi_image, device="cpu", conf=0.2)

        for result in roi_results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())
                
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                label = f"{roi_model.names[cls]} {conf:.2f}"
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color=(255,0,0), thickness=2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                
                # Crop the object from the frame
                needle_image = frame[y1:y2, x1:x2]

                #-- detect gauge
                needle_image = needle_preprocess_image(frame[y1:y2, x1:x2])
                needle_results = needle_model.predict(source=needle_image, device="cpu", conf=0.1)

                for g_result in needle_results:
                    g_boxes = g_result.boxes
                    for g_i, g_box in enumerate(g_boxes):
                        g_x1, g_y1, g_x2, g_y2 = g_box.xyxy[0]
                        g_x1, g_y1, g_x2, g_y2 = int(g_x1.item()), int(g_y1.item()), int(g_x2.item()), int(g_y2.item())
                        
                        g_conf  = g_box.conf[0].item()
                        g_cls   = int(g_box.cls[0].item())
                        g_label = f"{needle_model.names[g_cls]} {conf:.2f}"
                        
                        cv2.rectangle(frame[y1:y2, x1:x2], (g_x1, g_y1), (g_x2, g_y2), color=(0,255,0), thickness=2)
                        cv2.putText(frame[y1:y2, x1:x2], g_label, (g_x1, g_y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.imshow(win_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

        try:
            #-- handle close window
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                cap.release()
                cv2.destroyAllWindows()

                print("window closed")
                exit()
        except cv2.error:
            cap.release()
            cv2.destroyAllWindows()
            exit()
    
    cap.release()
    cv2.destroyAllWindows()