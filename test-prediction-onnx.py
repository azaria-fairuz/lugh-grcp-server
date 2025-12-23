import os
import cv2
import numpy as np
import onnxruntime as ort

def preprocess(frame):
    b, g, r = cv2.split(frame)
    imc = np.maximum(np.maximum(r, g), b)
    imc = imc.astype(np.uint8)
    imc = cv2.GaussianBlur(imc, (5,5), 0)

    atg = cv2.adaptiveThreshold(imc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    _, bw = cv2.threshold(atg, 130, 255, cv2.THRESH_BINARY_INV)

    kernel = np.ones((3,3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    rgb = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
    return rgb

def postprocess(outputs, conf_thresh=0.4):
    detections = outputs[0]
    detections = detections[0]
    results = []

    for det in detections:
        det = det.flatten()

        if len(det) == 5:
            x, y, w, h, conf = det
            if conf < conf_thresh:
                continue
            x1 = int(x - w/2)
            y1 = int(y - h/2)
            x2 = int(x + w/2)
            y2 = int(y + h/2)
            cls = 0
            results.append((cls, conf, x1, y1, x2, y2))

        elif len(det) == 6:
            x, y, w, h, obj_conf, class_score = det
            conf = obj_conf * class_score
            if conf < conf_thresh:
                continue
            cls = 0 if class_score < 0.5 else 1
            x1 = int(x - w/2)
            y1 = int(y - h/2)
            x2 = int(x + w/2)
            y2 = int(y + h/2)
            results.append((cls, conf, x1, y1, x2, y2))

        elif len(det) == 7:
            x, y, w, h, obj_conf, c0, c1 = det
            class_scores = np.array([c0, c1])
            cls = int(class_scores.argmax())
            conf = obj_conf * class_scores[cls]
            if conf < conf_thresh:
                continue
            x1 = int(x - w/2)
            y1 = int(y - h/2)
            x2 = int(x + w/2)
            y2 = int(y + h/2)
            results.append((cls, conf, x1, y1, x2, y2))

    return results

window_name = "GAUGE-DETECTION"
rtsp_url    = "rtsp://admin:Pdu12345678@191.101.190.233:8554/Streaming/Channels/101?rtsp_transport=tcp"

roi_session = ort.InferenceSession(os.path.join(os.getcwd(), "models", "gauge-roi.onnx"), providers=["CPUExecutionProvider"])
roi_input   = roi_session.get_inputs()[0].name
roi_output  = roi_session.get_outputs()[0].name

cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print("RTSP open failed")
    exit()

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    img = preprocess(frame)
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2,0,1))
    img = img[None, ...]

    outputs    = roi_session.run(None, {roi_input: img})
    detections = postprocess(outputs)

    H, W, _ = frame.shape
    for cls, conf, x1, y1, x2, y2 in detections:
        x1 = max(0, min(W-1, x1))
        y1 = max(0, min(H-1, y1))
        x2 = max(0, min(W-1, x2))
        y2 = max(0, min(H-1, y2))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{cls}:{conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    try:
        prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
        if prop < 1:
            break
    except cv2.error:
        break

cap.release()
cv2.destroyAllWindows()