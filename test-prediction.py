import os
import cv2
import time
import numpy as np
from ultralytics import YOLO

last_positions = {}

CLASS_TOP    = 0
CLASS_BOTTOM = 1
CLASS_CENTER = 2
CLASS_CIRCLE = 3

CALIBRATION = [
    {
        'max_value': 540,
        'unit': 'klb'
    },
    {
        'max_value': 235,
        'unit': 'kN'
    },
    {
        'max_value_1': 20,
        'max_value_2': 100,
        'start_angle': 24,
        'end_angle': 360,
        'unit': 'klb'
    },
    {
        'max_value_1': 10,
        'max_value_2': 45,
        'start_angle': 28,
        'end_angle': 360,
        'unit': 'kN'
    },
]

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

def needle_result(results):
    for r in results:
        plotted = r.plot()
        frame   = plotted.copy()

        if r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            clss  = r.boxes.cls.cpu().numpy()
            track_ids = r.boxes.id.cpu().numpy()

            tops, centers, circles = [], [], []

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                if clss[i] == CLASS_CENTER:
                    centers.append((cx, cy))

                elif clss[i] == CLASS_TOP:
                    top_id = track_ids[i]
                    reassigned = False
                    for prev_id, (px, py) in last_positions.items():
                        distance = np.linalg.norm([cx - px, cy - py])
                        if distance < 50:
                            top_id = prev_id
                            reassigned = True
                            break

                    last_positions[top_id] = (cx, cy)
                    tops.append((cx, cy, top_id))

                elif clss[i] == CLASS_CIRCLE:
                    radius = int(((x2 - x1) + (y2 - y1)) / 4)
                    circles.append((cx, cy, radius))

            for circle_cx, circle_cy, radius in circles:
                valid_centers = [(cx, cy) for cx, cy in centers if (cx-circle_cx)**2 + (cy-circle_cy)**2 <= radius**2]
                valid_tops = [(tx, ty, tid) for tx, ty, tid in tops if (tx-circle_cx)**2 + (ty-circle_cy)**2 <= radius**2]

                if len(valid_centers)==0 or len(valid_tops)==0:
                    continue

                for cx, cy in valid_centers:
                    needle_info = [(tx, ty, tid, np.linalg.norm([tx - cx, ty - cy])) for tx, ty, tid in valid_tops]
                    if len(needle_info) == 0:
                        continue  
                        
                    short_needle = min(needle_info, key=lambda x: x[3])
                    long_needle  = max(needle_info, key=lambda x: x[3])

                    short_tx, short_ty = short_needle[0], short_needle[1]
                    long_tx, long_ty   = long_needle[0], long_needle[1]
                    
                    if short_needle == long_needle:
                        stable_topids = [(short_tx, short_ty, 0)]
                    else:
                        stable_topids = [(short_tx, short_ty, 0), (long_tx, long_ty, 1)]

                    for tx, ty, stable_id in stable_topids:
                        lx = int(cx + (tx - cx) * 1.15)
                        ly = int(cy + (ty - cy) * 1.15)
                        cv2.line(frame, (cx, cy), (lx, ly), (0,0,255), 3)

                        raw_angle = np.degrees(np.arctan2(cy - ty, tx - cx)) 
                        raw_angle = (raw_angle + 360) % 360

                        if raw_angle <= 270:
                            angle = 270 - raw_angle
                        else:
                            angle = 630 - raw_angle

                        if stable_id == 0:
                            del c1, c2
                            c1 = angle / (360 / CALIBRATION[0]['max_value']) 
                            c2 = angle / (360 / CALIBRATION[1]['max_value']) 
                    
                        elif stable_id == 1:
                            del c1, c2
                            if raw_angle <= 90: 
                                an = 90 - raw_angle
                                c1 = (an / CALIBRATION[2]['start_angle']) * CALIBRATION[2]['max_value_1']
                                c2 = (an / CALIBRATION[3]['start_angle']) * CALIBRATION[3]['max_value_1']
                            else:
                                an = raw_angle - 90
                                c1 = (an / CALIBRATION[2]['end_angle']) * CALIBRATION[2]['max_value_2']
                                c2 = (an / CALIBRATION[3]['end_angle']) * CALIBRATION[3]['max_value_2']

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        texts = [
                            (f"angle : {raw_angle:.1f}", (lx+10, ly+20), 0.6),
                            (f"gauge_angle : {angle:.1f}", (lx+10, ly+45), 0.6)
                        ]
                        
                        for text, (x, y), scale in texts:
                            thickness = 1
                            text_size, _ = cv2.getTextSize(text, font, scale, thickness)
                            text_w, text_h = text_size
                        
                            cv2.rectangle(frame, (x, y - text_h - 2), (x + text_w, y + 2), (0, 255, 255), -1)
                            cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), thickness)

                        lines = [
                            (f"c1_value: {c1:.1f} {CALIBRATION[0]['unit']}", 0.6),
                            (f"c2_value: {c2:.1f} {CALIBRATION[1]['unit']}", 0.6)
                        ]
                        
                        if stable_id == 1:
                            lines.append((f"c3_value: {c1:.1f} {CALIBRATION[2]['unit']}", 0.6))
                            lines.append((f"c4_value: {c2:.1f} {CALIBRATION[3]['unit']}", 0.6))

                        y = 10
                        for line, scale in lines:
                            thickness = 1
                            (text_w, text_h), _ = cv2.getTextSize(line, font, scale, thickness)
                        
                            x = (frame.shape[1] - text_w) // 2
                        
                            cv2.rectangle(frame, (x - 5, y), (x + text_w + 5, y + text_h + 5), (0, 255, 255), -1)
                            cv2.putText(frame, line, (x, y + text_h), font, scale, (0, 0, 0), thickness)

                            y += text_h + 12

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